import os
import tempfile
import pandas as pd
import requests
import torch
import numpy as np  # For data cleaning and image processing
import io           # For reading uploaded file
import re           # For cleaning OCR text
import easyocr      # For OCR
import fitz
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, text
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
from langchain_community.embeddings import HuggingFaceEmbeddings
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

# Use HuggingFaceEmbeddings (LangChain wrapper around sentence-transformers)
# so it exposes .embed_query() / .embed_documents() that LangChain's Qdrant
# vector store and RetrievalQA expect. A raw SentenceTransformer only has
# .encode() and breaks RAG search.
embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
EMBEDDING_DIM = len(embedding_model.embed_query("dimension probe"))

# --- 4. Phase 1: Floorplan Model (YOLO+OCR) ---
model_cache = {}   # This will store our loaded models
ocr_reader = None  # Lazy-load the OCR model

def parse_floorplan(local_image_path: str, model_name: str = "best_1000.pt") -> dict:
    global model_cache, ocr_reader

    # Lazy-load YOLO model
    if model_name not in model_cache:
        model_path = os.path.join("/app", model_name)
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return {"error": f"Model file not found: {model_name}"}
        print(f"Loading floorplan model {model_name}...")
        model_cache[model_name] = YOLO(model_path)
    
    model = model_cache[model_name] # Use the selected model
    
    # Lazy-load OCR model
    if ocr_reader is None:
        print("Lazy loading OCR model (EasyOCR)...")
        ocr_reader = easyocr.Reader(['en'])
        print("OCR model loaded.")
    
    if not os.path.exists(local_image_path):
        print(f"Error: Image file not found at {local_image_path}")
        return {"error": f"Image file not found: {local_image_path}"}
    
    print(f"Parsing image: {local_image_path} with model: {model_name}")
    
    try:
        img_pil = Image.open(local_image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {local_image_path}: {e}")
        return {"error": f"Could not open image: {e}"}

    # Run YOLO detection
    results = model.predict(img_pil, imgsz=640, conf=0.25)
    result = results[0]

    counts = {"rooms": 0, "halls": 0, "kitchens": 0, "bathrooms": 0, "others" : 0}
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
                    else:
                        counts["others"] += 1
            

    json_output = {
        "rooms": counts["rooms"],
        "halls": counts["halls"],
        "kitchens": counts["kitchens"],
        "bathrooms": counts["bathrooms"],
        "other rooms": counts["others"]
    }
    
    return json_output

def parse_local_pdf(local_pdf_path: str) -> str:
    """Opens a local PDF file and extracts all text."""
    if not local_pdf_path or not os.path.exists(local_pdf_path):
        print(f"Warning: PDF file not found at {local_pdf_path}")
        return ""
    
    try:
        pdf_text = ""
        with fitz.open(local_pdf_path) as doc:
            for page in doc:
                pdf_text += page.get_text()
        
        print(f"Successfully parsed {len(pdf_text)} chars from {os.path.basename(local_pdf_path)}")
        return pdf_text
    
    except Exception as e:
        print(f"Warning: Could not parse PDF {local_pdf_path}. Error: {e}")
        return "" # Return empty string on failure

def fetch_and_parse_pdf(pdf_url: str) -> str:
    """Downloads a PDF from a URL and extracts all text."""
    if not pdf_url or not isinstance(pdf_url, str) or not pdf_url.lower().endswith('.pdf'):
        return "" # Return empty string if no valid PDF URL
    
    try:
        print(f"Fetching PDF from: {pdf_url}")
        response = requests.get(pdf_url, timeout=10) # 10 sec timeout
        response.raise_for_status() # Raise error if bad response (404, 500)
        
        pdf_text = ""
        # Open the PDF from in-memory bytes
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            for page in doc:
                pdf_text += page.get_text()
        
        print(f"Successfully parsed {len(pdf_text)} chars from {pdf_url}")
        return pdf_text
    
    except Exception as e:
        print(f"Warning: Could not parse PDF from {pdf_url}. Error: {e}")
        return "" # Return empty string on failure

# -----------------------------------------------------------------
# --- 5. THE LIFESPAN FIX ---
# We initialize all DB-dependent agents inside this startup event
# -----------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs ONCE, when the app starts up
    print("FastAPI is starting up, waiting for databases...")
    
    # 1. Create Qdrant Collection (idempotent — do NOT wipe data on restart)
    existing = {c.name for c in qdrant_client.get_collections().collections}
    if QDRANT_VECTOR_COLLECTION in existing:
        print(f"Qdrant collection '{QDRANT_VECTOR_COLLECTION}' already exists. Reusing.")
    else:
        qdrant_client.create_collection(
            collection_name=QDRANT_VECTOR_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print(f"Qdrant collection '{QDRANT_VECTOR_COLLECTION}' created.")

    # 2. Create PostgreSQL Tables
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully.")

    # 3. Initialize ALL database-dependent agents
    print("Initializing agents...")
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, api_key=GROQ_API_KEY)
    
    db = SQLDatabase(engine, include_tables=["properties"]) # <-- This connection is now DELAYED
    sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_agent = create_sql_agent(llm=llm, toolkit=sql_toolkit, agent_type="openai-tools", verbose=True)
    sql_search_tool = Tool(name="structured_property_search", func=sql_agent.invoke, description="Use to query the 'properties' table for properties based on price, location, rooms, etc.")
    
    vector_store = Qdrant(client=qdrant_client, collection_name=QDRANT_VECTOR_COLLECTION, embeddings=embedding_model)
    rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())
    rag_search_tool = Tool(name="unstructured_property_search", func=rag_chain.invoke, description="Use to search property descriptions for semantic info like 'family-friendly' or 'good view'.")
    
    @tool
    def generate_property_report(location: str = None, min_price: float = None, max_price: float = None, min_rooms: int = None) -> str:
        """
        Generates a summary report of properties matching search criteria.
        Use this when the user asks for a 'report', 'summary', or 'list' of properties.
        Arguments:
            location (str): The city or area to search in.
            min_price (float): The minimum price.
            max_price (float): The maximum price.
            min_rooms (int): The minimum number of rooms.
        """
        print(f"--- Report Generation Agent Triggered ---")
        db = SessionLocal()
        try:
            # Build the query dynamically based on provided arguments
            query = db.query(Property)
            filters = []
            
            if location:
                filters.append(Property.location.ilike(f"%{location}%"))
            if min_price is not None:
                filters.append(Property.price >= min_price)
            if max_price is not None:
                filters.append(Property.price <= max_price)
            if min_rooms is not None:
                filters.append(Property.rooms >= min_rooms)
            
            if filters:
                query = query.filter(*filters)
                
            properties = query.all()
            
            if not properties:
                return "I found no properties matching those criteria."

            # --- Create a Markdown report string ---
            report = f"# Property Report\n\n"
            report += f"I found {len(properties)} properties matching your criteria.\n\n---\n\n"
            
            for prop in properties:
                price_str = f"₹{prop.price:,.0f}" if prop.price is not None else "N/A"
                rooms_str = str(prop.rooms) if prop.rooms is not None else "N/A"
                bath_str = str(prop.bathrooms) if prop.bathrooms is not None else "N/A"
                if prop.description:
                    desc_str = prop.description[:150] + ("..." if len(prop.description) > 150 else "")
                else:
                    desc_str = "N/A"

                report += f"### {prop.title or 'Untitled property'}\n"
                report += f"* **Location:** {prop.location or 'N/A'}\n"
                report += f"* **Price:** {price_str}\n"
                report += f"* **Rooms:** {rooms_str}\n"
                report += f"* **Bathrooms:** {bath_str}\n"
                report += f"* **Description:** {desc_str}\n\n"
            
            print(f"Generated report with {len(properties)} properties.")
            return report
        
        except Exception as e:
            print(f"Error generating report: {e}")
            return "Sorry, I was unable to generate the report due to an error."
        finally:
            db.close()

    # --- Assemble Agent ---
    # --- Assemble Agent ---
    tools = [sql_search_tool, rag_search_tool, generate_property_report]
    
    # --- THIS IS THE FIX: A more robust system prompt ---
    system_prompt = """You are a specialized real-estate assistant.
Your goal is to answer questions about properties using the tools provided.
You can use the 'structured_property_search' tool for facts like price, location, and room counts.
You can use the 'unstructured_property_search' tool for more general questions about descriptions, neighborhoods, or report details.

If the user asks a question that is NOT related to real estate, properties, or your other tools,
and it is NOT a simple greeting (like 'hello'), you MUST respond with this exact sentence:
'I am a property information helper, I can't help you with this information.'

Do not answer general knowledge questions, write code, or discuss other topics.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),  # <-- Use the new, detailed prompt
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    # --- END OF FIX ---
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

@app.get("/health")
def health_check(request: Request):
    checks = {}
    healthy = True

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["postgres"] = True
    except Exception as e:
        print(f"Health check: Postgres connectivity failed: {e}")
        checks["postgres"] = False
        healthy = False

    try:
        qdrant_client.get_collections()
        checks["qdrant"] = True
    except Exception as e:
        print(f"Health check: Qdrant connectivity failed: {e}")
        checks["qdrant"] = False
        healthy = False

    checks["agent_executor"] = getattr(request.app.state, "agent_executor", None) is not None
    if not checks["agent_executor"]:
        healthy = False

    status_code = 200 if healthy else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "ok" if healthy else "unhealthy", "checks": checks},
    )

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

def _ingest_properties_sync(file_contents: bytes, model_name: str) -> dict:
    db = SessionLocal()
    try:
        df = pd.read_excel(io.BytesIO(file_contents))

        # --- Data Cleaning ---
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.replace({np.nan: None})
        # -----------------------

        print(f"Read {len(df)} rows from Excel. Cleaned price data.")
        
        qdrant_points = []
        ingested = 0
        
        for index, row in df.iterrows():
            # --- Use .get() for safety ---
            image_filename = row.get('image_file')
            certs_link_str = str(row.get('certificates', '')) # Get certs as string
            long_desc = row.get('long_description')
            price_val = row.get('price')
            title_val = row.get('title')
            location_val = row.get('location')
            
            if not image_filename:
                print(f"Skipping row {index}: No image filename.")
                continue

            # Construct the local path to the image
            local_image_path = os.path.join("/app/data/images", str(image_filename))
            
            floorplan_data = parse_floorplan(local_image_path, model_name)
            if floorplan_data.get("error"):
                print(f"Skipping row {index}: {floorplan_data['error']}")
                continue 

            # --- NEW PDF PARSING LOGIC ---
            report_text = ""
            if certs_link_str:
                links = certs_link_str.split('|') # Split by pipe
                for link in links:
                    if link and link.strip().lower().endswith('.pdf'):
                        print(f"Found PDF link: {link.strip()}")
                        pdf_path = os.path.join("/app/data/certificates", link.strip())
                        report_text += parse_local_pdf(pdf_path) + "\n\n"
            # ---------------------------------
            
            db_property = Property(
                title=title_val,
                description=long_desc,
                location=location_val,
                price=price_val,
                listing_date=row.get('listing_date'),
                certifications_link=certs_link_str,
                floorplan_image_url=image_filename,
                rooms=floorplan_data.get('rooms'),
                halls=floorplan_data.get('halls'),
                kitchens=floorplan_data.get('kitchens'),
                bathrooms=floorplan_data.get('bathrooms')
            )
            db.add(db_property)
            db.flush()  # populate db_property.id without committing yet
            sql_id = db_property.id
            
            # --- UPDATED TEXT FOR EMBEDDING ---
            text_to_embed = f"Title: {title_val}. Description: {long_desc}. Location: {location_val}. Reports: {report_text}"
            embedding = embedding_model.embed_query(text_to_embed)
            
            payload = {"text": text_to_embed, "property_id": sql_id}
            qdrant_points.append(PointStruct(id=sql_id, vector=embedding, payload=payload))
            ingested += 1

        db.commit()
        
        if qdrant_points:
            qdrant_client.upsert(
                collection_name=QDRANT_VECTOR_COLLECTION,
                points=qdrant_points,
                wait=True
            )
        
        return {"status": "success", "message": f"Successfully ingested {ingested} properties."}
    
    except KeyError as e:
        db.rollback()
        print(f"Ingestion error: Missing column {e}")
        raise HTTPException(status_code=400, detail=f"Missing column in Excel file: {e}")
    except Exception as e:
        db.rollback()
        import traceback
        traceback.print_exc()
        print(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion Error: {str(e)}")
    finally:
        db.close()

@app.post("/ingest")
async def ingest_properties(file: UploadFile = File(...), model_name: str = "best_1000.pt"):
    # Read the file into an in-memory bytes buffer (async I/O), then hand the
    # CPU-heavy parsing/ingest work (pandas, YOLO, EasyOCR, DB writes) off to a
    # worker thread so it doesn't block the event loop for the whole request.
    file_contents = await file.read()
    return await run_in_threadpool(_ingest_properties_sync, file_contents, model_name)

@app.post("/parse-floorplan-debug")
async def parse_floorplan_debug(file: UploadFile = File(...), model_name: str = "best_1000.pt"):
    contents = await file.read()

    # Never trust the client-supplied filename as a path (path traversal) —
    # write to a generated temp path instead, and always clean it up.
    fd, file_path = tempfile.mkstemp(dir="/tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(contents)
        data = await run_in_threadpool(parse_floorplan, file_path, model_name)
    finally:
        os.remove(file_path)

    return data