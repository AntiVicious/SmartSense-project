import os
import pandas as pd
import io  # <-- ADD THIS LINE
import requests
import torch
import json
import numpy as np  # <-- The missing numpy import
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from contextlib import asynccontextmanager  # <-- For the lifespan fix
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from ultralytics import YOLO

# --- Agent & Chat Imports (The Stable Pinned Version) ---
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
from langchain_core.messages import HumanMessage, AIMessage  # <-- Correct 'AIMessage'
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit  # <-- Correct imports

# ... (all your imports) ...

# --- 1. Environment & Config (Using Docker Service Names) ---
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")

# --- THIS IS THE FIX ---
# We must connect to the Docker *service name*, not localhost
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres-db") 
# -----------------------

DB_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}/{POSTGRES_DB}?sslmode=disable"

# --- THIS IS THE FIX ---
# We must connect to the Docker *service name*, not localhost
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant-db") 
# -----------------------

QDRANT_VECTOR_COLLECTION = "properties"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ... (rest of your file is correct) ...
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
qdrant_client = QdrantClient(host=QDRANT_HOST, port=6333)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()
try:
    qdrant_client.recreate_collection(
        collection_name=QDRANT_VECTOR_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
except Exception as e:
    print(f"Qdrant collection might already exist: {e}")

# --- 4. Phase 1: Floorplan Model (REAL) ---
model = None  # Lazy-load the model
def parse_floorplan(local_image_path: str) -> dict: # <-- Takes a local path
    global model
    if model is None:
        print("Lazy loading floorplan model...")
        model = YOLO("best.pt")
    
    # Check if file exists before predicting
    if not os.path.exists(local_image_path):
        print(f"Error: Image file not found at {local_image_path}")
        return {"error": f"Image file not found: {local_image_path}"}
    
    results = model.predict(local_image_path, imgsz=640, conf=0.25) # <-- Predicts on path
    result = results[0]
    json_output = {"rooms": 0, "halls": 0, "kitchens": 0, "bathrooms": 0, "rooms_detail": []}
    class_names = result.names 
    room_details_map = {}
    if result.masks is not None:
        for i in range(len(result.masks)):
            class_id = int(result.boxes.cls[i])
            label = class_names[class_id]
            mask = result.masks.data[i].cpu().numpy()
            area = float(np.sum(mask))  # <-- Using np.sum()
            if label in json_output:
                json_output[label] += 1
            if label not in room_details_map:
                room_details_map[label] = {"count": 0, "total_area": 0.0}
            room_details_map[label]["count"] += 1
            room_details_map[label]["total_area"] += area
    for label, data in room_details_map.items():
        json_output["rooms_detail"].append({
            "label": label, "count": data["count"], "approx_area": data["total_area"]
        })
    return json_output

# -----------------------------------------------------------------
# --- 5. THE LIFESPAN FIX ---
# We initialize all DB-dependent agents inside this startup event
# -----------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs ONCE, when the app starts up
    print("FastAPI is starting up...")

    # 1. Create tables
    try:
        Base.metadata.create_all(bind=engine)
        print("Tables created successfully.")
    except Exception as e:
        print(f"Error creating tables: {e}")
        # In a real app, you might want to raise this
    
    # 2. Initialize ALL database-dependent agents
    print("Initializing agents...")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=GROQ_API_KEY)
    
    # --- SQL Agent ---
    db = SQLDatabase(engine, include_tables=["properties"]) # This connection is now DELAYED    
    sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_agent = create_sql_agent(llm=llm, toolkit=sql_toolkit, agent_type="openai-tools", verbose=True)
    sql_search_tool = Tool(name="structured_property_search", func=sql_agent.invoke, description="Use to query database for properties based on price, location, rooms, etc.")
    
    # --- RAG Agent ---
    vector_store = Qdrant(client=qdrant_client, collection_name=QDRANT_VECTOR_COLLECTION, embeddings=embedding_model)
    rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())
    rag_search_tool = Tool(name="unstructured_property_search", func=rag_chain.invoke, description="Use to search descriptions for semantic info like 'family-friendly' or 'good view'.")
    
    # --- Mock Tools ---
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
app = FastAPI(title="Real Estate Search API", lifespan=lifespan)

# --- THIS IS THE FIX for the 404 Error ---
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Backend is running!"}
# ----------------------------------------

# --- 7. API Endpoints ---
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
        
        # --- THIS IS THE FIX ---
        # 1. Force 'price' to be numeric. Any strings (like 'fire-safety.pdf')
        #    will be converted to NaN (Not a Number).
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        # 2. Convert all NaN values to None, which SQL can handle as NULL.
        df = df.replace({np.nan: None})
        # -----------------------

        print(f"Read {len(df)} rows from Excel. Cleaned price data.")
        
        qdrant_points = []
        point_id = 1
        
        for index, row in df.iterrows():
            # --- Get the correct, cleaned data ---
            image_filename = row.get('image_file')
            certs_link = row.get('certificates')
            long_desc = row.get('long_description')
            price_val = row.get('price') # This is now clean (a number or None)
            
            # --- Build the local image path ---
            local_image_path = os.path.join("/app/data/images", str(image_filename))
            
            # --- Parse the floorplan ---
            floorplan_data = parse_floorplan(local_image_path)
            if floorplan_data.get("error"):
                print(f"Skipping row {index}: {floorplan_data['error']}")
                continue 

            db_property = Property(
                title=row.get('title'),
                description=long_desc,
                location=row.get('location'),
                price=price_val, # <-- Use the cleaned price
                listing_date=row.get('listing_date'),
                certifications_link=certs_link,
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
    data = parse_floorplan(file_path)
    return data