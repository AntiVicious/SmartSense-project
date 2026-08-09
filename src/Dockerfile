FROM python:3.10-slim
WORKDIR /app

# --- 1. Install System Dependencies ---
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# --- 2. Install HEAVY Python Dependencies ---
COPY requirements-heavy.txt .  
RUN pip install --no-cache-dir -r requirements-heavy.txt

# --- 3. Install LIGHT Python Dependencies ---
COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt

# --- 4. Copy the App Code ---
# Preserved as a package (src/api.py, src/config.py, ...) rather than
# flattened into /app — api.py uses relative imports (from .config import
# ...), which only resolve if src stays a real package both here and in
# local dev, where `python -c "import src.api"` is run from the repo root.
# .dockerignore excludes src/*.pt, so this never bakes in a stale local
# copy of the weights — the fetch step below is the only source.
COPY src/ ./src/

# --- 4b. Fetch model weights ---
# best_1000.pt isn't committed to git (see README) — it's published as a
# GitHub release asset instead. Pinned to an exact tag + checksum so a
# build never silently picks up a different file.
ARG MODEL_WEIGHTS_URL=https://github.com/AntiVicious/SmartSense-project/releases/download/model-weights-v1/best_1000.pt
ARG MODEL_WEIGHTS_SHA256=742fd385ac46ba7dea4c8a36c38c8014fe3b7f3c5bf9a8de9efb2947daea9545
RUN python -c "import urllib.request; urllib.request.urlretrieve('${MODEL_WEIGHTS_URL}', '/app/src/best_1000.pt')" \
    && echo "${MODEL_WEIGHTS_SHA256}  /app/src/best_1000.pt" | sha256sum -c -

# --- 5. Copy the Data ---
COPY data /app/data

# --- 6. FIX line endings & make executable ---
RUN sed -i 's/\r$//' ./src/run.sh
RUN chmod +x ./src/run.sh

# --- 7. Run the App ---
CMD ["./src/run.sh"]