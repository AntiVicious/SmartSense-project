
# 🏡 Real Estate Search Engine – AI/ML Case Study

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-CV%20Model-orange)
![LangChain](https://img.shields.io/badge/LangChain-MultiAgent-yellow)

A **full-stack, AI-powered Real Estate Search Engine** built for the **Smart Sense AI/ML Case Study**.

It integrates:
- 🧠 A **custom-trained computer vision model** to parse floorplans  
- 🧩 A **dual-database ETL pipeline** (PostgreSQL + Qdrant)  
- 🤖 A **multi-agent chatbot** that supports natural language property queries  
- 🐳 **Full containerization** with Docker for seamless deployment  

---

## 🚀 How to Run

### 🧱 1. Prerequisites
- **Docker & Docker Compose** (installed and running)
- **Git**
- **API Keys**
  - [Groq](https://groq.com/) → for LLM  
  - [Tavily](https://tavily.com/) → for web search

---

### ⚙️ 2. Setup Instructions

#### Clone the Repository
```bash
git clone [Your-Repo-URL]
cd real-estate-search
```

#### Set Up Environment File
```bash
cp .env.example .env
```

Edit `.env` with your API keys:
```bash
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly_...

# Default database credentials (optional to change)
POSTGRES_USER=myuser
POSTGRES_PASSWORD=mypassword
POSTGRES_DB=real_estate_db
```

#### Add Project Data
| Type | Folder | Notes |
|------|---------|-------|
| 🧾 Excel/CSV | `data/` | Place `Property_list.xlsx` or `.csv` |
| 🏠 Images | `data/images/` | Floorplan images |
| 📜 PDFs | `data/pdfs/` | Certificate PDFs (e.g., `fire-safety.pdf`) |
| 🧠 Model | `src/` | Trained YOLO model (`best.pt`, `best_300.pt`, etc.) |

#### Run the App
```bash
docker-compose up --build
```
> 🕒 First build: 30–45 mins (heavy deps)  
> ⚡ Subsequent rebuilds: 1–2 mins only

#### Access
- **Frontend UI:** [http://localhost:8501](http://localhost:8501)  
- **Backend Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📁 Project Structure
```
real-estate-search/
├── src/
│   ├── api.py               # FastAPI backend (agents, ingestion logic)
│   ├── main.py              # Streamlit frontend (UI)
│   ├── run.sh               # Runs FastAPI + Streamlit in one container
│   ├── Dockerfile
│   ├── best_300.pt          # Trained model (300 epochs)
│   └── best_1000.pt         # Trained model (1000 epochs)
│
├── data/
│   ├── Property_list.xlsx
│   ├── images/
│   └── pdfs/
│
├── .env
├── .env.example
├── docker-compose.yml
├── requirements.txt
└── requirements-heavy.txt
```

---

### ✅ Summary
This Real Estate Search Engine demonstrates:
- 🧠 **AI-driven insights** via CV + OCR + NLP  
- 🧩 **Hybrid data pipeline**: structured + unstructured  
- 🤖 **LLM-powered multi-agent chatbot**  
- 🐳 **Production-grade Docker deployment** (FastAPI + Streamlit)

---

**Phase 1: Floorplan Computer Vision Model**
The case study required training a CV model from scratch to extract room counts.

*Key Decision: 2-Stage (YOLOv8 + OCR) Pipeline*
Challenge: The provided annotations.coco.json file did not contain labels for room types (e.g., "kitchen", "bathroom"). Instead, it contained bounding boxes for text elements like room_name and room_dim.

Solution: I implemented a 2-stage pipeline to solve this:

Stage 1 (Detection): A YOLOv8s (small) model was trained from scratch (using its .yaml architecture file) on the annotations.coco.json data. This model's only job is to find the bounding boxes for room_name text on the floorplan.

Stage 2 (Recognition): For each room_name box detected, the api.py script crops that part of the image and feeds it to an easyocr model. This OCR model reads the text (e.g., "KITCHEN", "BATH") and classifies it, allowing us to get the counts for the required categories.

Metrics: The YOLOv8 model was trained on an 80/20 split of the labeled data and achieved a final mAP@.5 of [Your mAP Score Here] after 300 epochs.

This approach not only met the "train from scratch" requirement using PyTorch (YOLO is a PyTorch framework) but also demonstrated a robust, real-world solution to a "dirty data" problem.

**Phase 2: Data Ingestion Pipeline**
The /ingest endpoint on the FastAPI server handles the full ETL process:

Extract: The user-uploaded Excel file is read into memory using pandas.

Transform:

Data Cleaning: The price column is converted to numeric, forcing errors (like text) into NULL values.

Image Parsing: For each row, the image_file name is used to find the local file (e.g., /app/data/images/file.jpg). This file is passed to the 2-stage parse_floorplan function to get the room counts.

PDF Parsing: The certificates column is parsed. Each PDF filename is found in /app/data/pdfs/, and PyMuPDF extracts its text.

Load:

PostgreSQL: All structured data (title, price, location, room counts, etc.) is saved to the properties table.

Qdrant: A single document (containing the title, description, and extracted PDF text) is vectorized using sentence-transformers and indexed in the vector database.

**Phase 3: Multi-Agent Chatbot**
The chatbot is a Tool-Calling Agent built with LangChain, which acts as the Query Router and Task Planner. It uses the Groq llama-3.1-70b-versatile LLM to decide which tool to use.

The agent has access to 5 specialized tools:

structured_property_search (Real): An SQL Agent that connects to the PostgreSQL database to answer factual questions (e.g., "find 3-bedroom houses under 80 Lakhs").

unstructured_property_search (Real): A RAG Agent that connects to Qdrant to answer semantic questions (e.g., "find properties with a gym and good security").

web_researcher (Real): A tool using the Tavily API to get live web data (e.g., "what's the current market trend in Hyderabad?").

generate_property_report (Real): A custom tool that queries the SQL database to generate a formatted markdown report of properties.

renovation_estimator (Mock): A placeholder tool that returns a JSON object with a mock cost.

Memory: Conversational memory is implemented by passing the st.session_state chat history to the agent with each turn.

**Phase 4: UI & Deployment**
Architecture Decision: Single Container, Dual Process
Problem: Early prototypes with separate frontend and backend containers suffered from complex and unreliable Docker networking issues ("Connection Refused", "Host not found").

Solution: The final architecture uses a single app container. A run.sh script first launches the FastAPI server in the background and then launches the Streamlit app.

Benefits: This is extremely robust. The Streamlit app can now reliably connect to the FastAPI API at http://localhost:8000 as if it were a local process, completely eliminating all Docker networking problems.

Race Condition Solution
Problem: The app container would start faster than the postgres-db container, causing the api.py script to crash with Connection refused.

Solution:

A healthcheck was added to the postgres-db service in docker-compose.yml to check when the database is ready.

The app service was given a depends_on condition to wait for postgres-db to be service_healthy.

All database-dependent code in api.py (agent creation, table creation) was moved into a FastAPI lifespan event, ensuring it only runs after the database connection is guaranteed.

Build Optimization
The app/Dockerfile is optimized to cache the heavy, 30-minute installation (PyTorch, Ultralytics, etc.) in a separate layer using requirements-heavy.txt.

Changes to the application code (api.py, main.py) or light requirements (requirements.txt) only trigger a fast, 1-2 minute rebuild.

---


