#!/bin/sh

# Start the FastAPI backend server (api.py) in the background
python -m uvicorn api:app --host 0.0.0.0 --port 8000 &

# Start the Streamlit frontend (main.py) in the foreground
python -m streamlit run main.py --server.port 8501 --server.address 0.0.0.0