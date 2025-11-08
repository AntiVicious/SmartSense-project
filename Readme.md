# Real Estate Search Engine - Case Study

This project is a solution for the Smart Sense AI/ML case study. It implements a full-stack application with a computer vision model, data ingestion pipeline, and a multi-agent chatbot.

## Features
* **Phase 1: CV Model:** A YOLOv8 model trained to detect rooms, kitchens, etc., from floorplans.
* **Phase 2: Data Pipeline:** An ETL process to ingest Excel data, parse images, and load into PostgreSQL and Qdrant.
* **Phase 3: Agentic Chatbot:** A multi-agent system (SQL, RAG) to answer user queries.
* **Phase 4: UI/Backend:** A FastAPI backend and Streamlit frontend.

## How to Run
1.  Ensure Docker and Docker Compose are installed.
2.  Place your Groq API key in a `.env` file (see `.env.example`).
3.  Run the command: `docker-compose up --build`
4.  The application will be available at `http://localhost:8501`.

## Project Structure
* `data/`: Contains the source Excel/CSV file and floorplan images.
* `src/`: Contains all application source code (FastAPI, Streamlit, model files).
* `notebooks/`: (Optional) Contains the `train.ipynb` used for model training.
* `docker-compose.yml`: Defines all services.
* `requirements.txt`: Python dependencies.