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
COPY src/ ./src/

# --- 5. Copy the Data ---
COPY data /app/data

# --- 6. FIX line endings & make executable ---
RUN sed -i 's/\r$//' ./src/run.sh
RUN chmod +x ./src/run.sh

# --- 7. Run the App ---
CMD ["./src/run.sh"]