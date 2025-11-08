import os
import pandas as pd
import requests
import torch
import json
import numpy as np  # For data cleaning and image processing
import io           # For reading uploaded file
import re           # For cleaning OCR text
import easyocr      # For OCR
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from contextlib import asynccontextmanager
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from ultralytics import YOLO
from PIL import Image

# --- Agent & Chat Imports (Stable Pinned Version) ---
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor
from langchain_community.vectorstores import Qdrant
from langchain.tools import Tool, tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.chains import RetrievalQA
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit

# --- 1. Environment & Config (Using Docker Service Names) ---
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres-db") # Use service name
DB_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}/{POSTGRES_DB}?sslmode=disable"

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant-db") # Use service name
QDRANT_VECTOR_COLLECTION = "properties"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- 2. Database Setup (PostgreSQL) ---
Base = declarative_base()
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Property(Base):
    __tablename__ = "properties"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(Text)
    location = Column(String)
    price = Column(Float)
    listing_date = Column(String)
    certifications_link = Column(String)
    floorplan_image_url = Column(String)
    rooms = Column(Integer)
    halls = Column(Integer)
    kitchens = Column(Integer)
    bathrooms = Column(Integer)

# --- 3. Database Setup (Qdrant) ---
# We initialize clients here, but only connect/create tables in the lifespan.
qdrant_client = QdrantClient(host=QDRANT_HOST, port=6333)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()

# --- 4. Phase 1: Floorplan Model (YOLO+OCR) ---
yolo_model = None  # Lazy-load the YOLO model
ocr_reader = None  # Lazy-load the OCR model

def parse_floorplan(local_image_path: str) -> dict:
    global yolo_model, ocr_reader

    # Lazy-load YOLO model
    if yolo_model is None:
        print("Lazy loading floorplan model (YOLO)...")
        yolo_model = YOLO("best.pt")  # Assumes best.pt is in /app/
    
    # Lazy-load OCR model
    if ocr_reader is None:
        print("Lazy loading OCR model (EasyOCR)...")
        ocr_reader = easyocr.Reader(['en'])
        print("OCR model loaded.")
    
    if not os.path.exists(local_image_path):
        print(f"Error: Image file not found at {local_image_path}")
        return {"error": f"Image file not found: {local_image_path}"}
    
    print(f"Parsing image: {local_image_path}")
    
    try:
        img_pil = Image.open(local_image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {local_image_path}: {e}")
        return {"error": f"Could not open image: {e}"}

    # Run YOLO detection
    results = yolo_model.predict(img_pil, imgsz=640, conf=0.25)
    result = results[0]

    counts = {"rooms": 0, "halls": 0, "kitchens": 0, "bathrooms": 0}
    room_details = []
    class_names = result.names 
    
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = class_names[class_id]
            
            if label == 'room_name':
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords
                
                cropped_img_pil = img_pil.crop((x1, y1, x2, y2))
                cropped_img_np = np.array(cropped_img_pil)
                
                ocr_result_list = ocr_reader.readtext(cropped_img_np, detail=0)
                
                if ocr_result_list:
                    detected_text = " ".join(ocr_result_list).lower()
                    detected_text = re.sub(r'[^a-z\s]', '', detected_text).strip()
                    print(f"  > Detected label '{label}' -> OCR Text: '{detected_text}'")
                    
                    # --- Simple classification based on text ---
                    if "ki" in detected_text:
                        counts["kitchens"] += 1
                    elif "bath" in detected_text or "wc" in detected_text or "wash" in detected_text or "toi" in detected_text or "powder" in detected_text:
                        counts["bathrooms"] += 1
                    elif "hall" in detected_text or "liv" in detected_text or "great" in detected_text:
                        counts["halls"] += 1
                    elif "bed" in detected_text or "room" in detected_text or "br" in detected_text:
                        # This is a general "room", e.g., bedroom
                        counts["rooms"] += 1
            
            # This logic is for the 'rooms_detail' bonus
            if label == 'room_dim':
                coords = box.xyxy[0].cpu().numpy()
                width = float(coords[2] - coords[0])
                height = float(coords[3] - coords[1])
                area = float(width * height)
                room_details.append({"label": "room_dimension", "approx_area": area})

    json_output = {
        "rooms": counts["rooms"],
        "halls": counts["halls"],
        "kitchens": counts["kitchens"],
        "bathrooms": counts["bathrooms"],
        "rooms_detail": room_details
    }
    
    return json_output

# -----------------------------------------------------------------
# --- 5. THE LIFESPAN FIX ---
# We initialize all DB-dependent agents inside this startup event
# -----------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs ONCE, when the app starts up
    print("FastAPI is starting up, waiting for databases...")
    
    # 1. Create Qdrant Collection
    try:
        qdrant_client.recreate_collection(
            collection_name=QDRANT_VECTOR_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print("Qdrant collection created.")
    except Exception as e:
        print(f"Qdrant collection might already exist: {e}")

    # 2. Create PostgreSQL Tables
    try:
        Base.metadata.create_all(bind=engine)
        print("Tables created successfully.")
    except Exception as e:
        print(f"Error creating tables: {e}")
    
    # 3. Initialize ALL database-dependent agents
    print("Initializing agents...")
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0, api_key=GROQ_API_KEY)
    
    db = SQLDatabase(engine, include_tables=["properties"]) # <-- This connection is now DELAYED
    sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_agent = create_sql_agent(llm=llm, toolkit=sql_toolkit, agent_type="openai-tools", verbose=True)
    sql_search_tool = Tool(name="structured_property_search", func=sql_agent.invoke, description="Use to query the 'properties' table for properties based on price, location, rooms, etc.")
    
    vector_store = Qdrant(client=qdrant_client, collection_name=QDRANT_VECTOR_COLLECTION, embedding_function=embedding_model) # <-- Use embedding_function
    rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())
    rag_search_tool = Tool(name="unstructured_property_search", func=rag_chain.invoke, description="Use to search property descriptions for semantic info like 'family-friendly' or 'good view'.")
    
    @tool
    def renovation_estimator(property_details: str) -> str:
        """Estimates renovation cost. Mock tool."""
        return json.dumps({"estimated_cost_lakhs": 5, "note": "This is a mock estimate"})
    @tool
    def web_researcher(query: str) -> str:
        """Performs web research. Mock tool."""
        return json.dumps({"summary": "According to web research, the neighborhood is safe.", "note": "This is a mock result"})

    # --- Assemble Agent ---
    tools = [sql_search_tool, rag_search_tool, renovation_estimator, web_researcher]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful real-estate assistant. Route to the correct tool."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    llm_with_tools = llm.bind_tools(tools)
    main_agent = (
        {"input": lambda x: x["input"], "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]), "chat_history": lambda x: x["chat_history"],}
        | prompt | llm_with_tools | OpenAIToolsAgentOutputParser()
    )
    
    # 3. Store the final agent on the app.state
    app.state.agent_executor = AgentExecutor(agent=main_agent, tools=tools, verbose=True)
    
    print("--- FastAPI is ready and agents are initialized! ---")
    
    yield
    
    print("FastAPI is shutting down.")

# --- 6. FastAPI App Definition ---
app = FastAPI(title="Real-Estate API", lifespan=lifespan)

# --- 7. API Endpoints ---

@app.get("/")
def read_root():
    """Root endpoint for health checks."""
    return {"status": "ok", "message": "Backend is running!"}

class ChatRequest(BaseModel):
    query: str
    history: list[tuple[str, str]] = []

@app.post("/chat")
async def chat_endpoint(request: ChatRequest, fast_api_request: Request):
    # Get the agent from the app state
    agent = fast_api_request.app.state.agent_executor
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized.")

    chat_history = []
    for user_msg, ai_msg in request.history:
        chat_history.append(HumanMessage(content=user_msg))
        chat_history.append(AIMessage(content=ai_msg))
    
    try:
        # Run agent
        response = await agent.ainvoke({"input": request.query, "chat_history": chat_history})
        return {"status": "success", "response": response['output']}
    except Exception as e:
        print(f"Agent execution error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent Error: {e}")

@app.post("/ingest")
async def ingest_properties(file: UploadFile = File(...)):
    db = SessionLocal()
    try:
        # Read the file into an in-memory bytes buffer
        file_contents = await file.read()
        df = pd.read_excel(io.BytesIO(file_contents))
        
        # --- Data Cleaning ---
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.replace({np.nan: None})
        # -----------------------

        print(f"Read {len(df)} rows from Excel. Cleaned price data.")
        
        qdrant_points = []
        point_id = 1
        
        for index, row in df.iterrows():
            # --- Use .get() for safety ---
            image_filename = row.get('image_file')
            certs_link = row.get('certificates')
            long_desc = row.get('long_description')
            price_val = row.get('price')
            
            if not image_filename:
                print(f"Skipping row {index}: No image filename.")
                continue

            # Construct the local path to the image
            local_image_path = os.path.join("/app/images", str(image_filename))
            
            floorplan_data = parse_floorplan(local_image_path)
            if floorplan_data.get("error"):
                print(f"Skipping row {index}: {floorplan_data['error']}")
                continue 

            db_property = Property(
                title=row.get('title'),
                description=long_desc,
                location=row.get('location'),
                price=price_val,
                listing_date=row.get('listing_date'),
                certifications_link=str(certs_link) if certs_link else None,
                floorplan_image_url=image_filename,
                rooms=floorplan_data.get('rooms'),
                halls=floorplan_data.get('halls'),
                kitchens=floorplan_data.get('kitchens'),
                bathrooms=floorplan_data.get('bathrooms')
            )
            db.add(db_property)
            
            text_to_embed = f"Title: {row.get('title')}. Description: {long_desc}. Location: {row.get('location')}"
            embedding = embedding_model.encode(text_to_embed).tolist()
            payload = {"text": text_to_embed, "property_id": point_id}
            qdrant_points.append(PointStruct(id=point_id, vector=embedding, payload=payload))
            point_id += 1

        db.commit()
        qdrant_client.upsert(collection_name=QDRANT_VECTOR_COLLECTION, points=qdrant_points, wait=True)
        return {"status": "success", "message": f"Successfully ingested {point_id - 1} properties."}
    except KeyError as e:
        print(f"Ingestion error: Missing column {e}")
        raise HTTPException(status_code=400, detail=f"Missing column in Excel file: {e}")
    except Exception as e:
        db.rollback()
        print(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion Error: {e}")
    finally:
        db.close()

@app.post("/parse-floorplan-debug")
async def parse_floorplan_debug(file: UploadFile = File(...)):
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # We must use the local filesystem path
    data = parse_floorplan(file_path)
    return data