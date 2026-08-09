#!/bin/sh

# Start the FastAPI backend server (src/api.py) in the background.
# Run from /app (WORKDIR) so `src` resolves as a package — api.py uses
# relative imports (from .config import ...) that require it.
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 &

# Start the Streamlit frontend (src/main.py) in the foreground
python -m streamlit run src/main.py --server.port 8501 --server.address 0.0.0.0