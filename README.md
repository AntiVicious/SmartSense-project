# SmartSense — AI-Powered Real Estate Search

A backend that ingests property listings (spreadsheet + floorplan images + PDF
certificates), automatically counts rooms from the floorplans using a
custom-trained computer vision model, indexes everything into both a
relational and a vector database, and answers natural-language questions
about the resulting portfolio through a multi-agent chatbot.

- **Floorplan understanding**: a YOLOv8 model (trained from scratch) detects
  room-name labels on a floorplan image; EasyOCR reads each one and a
  classifier buckets it into rooms / halls / kitchens / bathrooms.
- **Dual-database ingestion**: structured fields go to PostgreSQL, a text
  embedding of the listing (description + location + certificate text) goes
  to Qdrant.
- **Multi-agent chat**: a LangChain tool-calling agent routes each question
  to a SQL agent (structured queries), a RAG chain over Qdrant (semantic
  search), or a custom report generator.
- **FastAPI + Streamlit**, containerized with Docker Compose.

Built with: Python 3.10 · FastAPI · Streamlit · PostgreSQL · Qdrant · YOLOv8
(Ultralytics) · EasyOCR · LangChain · Groq

---

## How to Run

### 1. Prerequisites

- Docker & Docker Compose
- Git
- A [Groq](https://groq.com/) API key

### 2. Clone and configure

```bash
git clone [Your-Repo-URL]
cd SmartSense-project
cp .env.example .env
```

Edit `.env`:

```bash
GROQ_API_KEY=gsk_...

# Database credentials (defaults are fine for local use)
POSTGRES_USER=myuser
POSTGRES_PASSWORD=mypassword
POSTGRES_DB=real_estate_db
```

### 3. Add your data

| Type | Folder | Notes |
|---|---|---|
| Excel/CSV | `data/` | e.g. `Property_list.xlsx` |
| Floorplan images | `data/images/` | referenced by the `image_file` column |
| Certificate PDFs | `data/certificates/` | referenced by the `certificates` column |
| Model weights | `src/best_1000.pt` | fetched automatically by Docker — see [Model weights](#model-weights) below |

### 4. Run it

```bash
docker compose up --build
```

First build takes 30–45 minutes (PyTorch, Ultralytics, EasyOCR, etc.); the
heavy dependency layer is cached separately from application code, so
subsequent rebuilds after a code change take 1–2 minutes.

- **UI:** http://localhost:8501
- **API docs:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health

---

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Trivial liveness ping |
| `/health` | GET | Checks Postgres, Qdrant, and agent initialization; 503 if any fail |
| `/chat` | POST | `{"query": str, "history": [[user, assistant], ...]}` → agent response |
| `/ingest` | POST | Upload an Excel/CSV file; parses floorplans, writes to Postgres + Qdrant |
| `/parse-floorplan-debug` | POST | Upload a single floorplan image, get back the room-count JSON |

---

## Project Structure

```
SmartSense-project/
├── src/
│   ├── api.py              # FastAPI app: routes + lifespan wiring only
│   ├── config.py           # pydantic-settings Settings (fails loudly on missing env vars)
│   ├── db.py                # lazy engine/Qdrant-client/embedder factories
│   ├── models.py            # SQLAlchemy Property model + request schemas
│   ├── floorplan.py         # YOLO + EasyOCR floorplan parsing, room classification
│   ├── documents.py         # PDF text extraction
│   ├── agents.py            # LangChain tool definitions + agent assembly
│   ├── ingest.py            # Excel -> Postgres + Qdrant ingestion pipeline
│   ├── main.py               # Streamlit UI
│   ├── Dockerfile             # fetches best_1000.pt from a GitHub release at build time
│   └── run.sh                 # launches FastAPI + Streamlit in one container
│
├── tests/                    # pytest-discoverable; see Development below
├── notebooks/
│   └── train.ipynb           # YOLO training notebook (not part of the runtime image)
├── data/
│   ├── Property_list.xlsx
│   ├── images/
│   └── certificates/
│
├── .github/workflows/ci.yml  # lint + test on push to main
├── docker-compose.yml
├── requirements.txt           # light deps (fastapi, langchain, ...)
├── requirements-heavy.txt     # heavy deps (torch, ultralytics, easyocr, ...)
└── requirements-dev.txt       # black, flake8, pytest
```

---

## Design Notes

**Single container, dual process.** Separate frontend/backend containers hit
unreliable Docker networking during early development ("Connection
Refused", "Host not found"). `run.sh` backgrounds the FastAPI server and
runs Streamlit in the foreground of the same container, so the UI talks to
the API over `localhost` like a local process. The tradeoff — if uvicorn
crashes, the container looks alive because Streamlit still is — is a known
issue tracked for a future split into two Compose services.

**Startup ordering.** The `app` container previously raced `postgres-db` and
crashed with `Connection refused`. Fixed with a Postgres healthcheck plus a
`depends_on: condition: service_healthy` in `docker-compose.yml`, and by
moving all database-dependent construction (engine, Qdrant client, agent
assembly) into FastAPI's `lifespan`, which only runs once the container
network is up.

**Lazy construction for testability.** `config.py`, `db.py`, and friends
build nothing at import time — the engine, Qdrant client, and embedding
model are constructed once in `lifespan` and handed to routes via FastAPI
`Depends`. This is what makes `import src.api` (and the test suite) work
without a network connection, a model download, or a live database.

**Build layering.** `requirements-heavy.txt` (PyTorch, Ultralytics, EasyOCR —
the ~30 minute part) is installed in its own Docker layer, separate from
`requirements.txt` and the application code. Changing `api.py` or a light
dependency triggers a 1–2 minute rebuild, not a 30-minute one.

---

## Floorplan CV Model

The case study's annotations (`annotations.coco.json`) didn't label room
types directly — they were bounding boxes for text elements (`room_name`,
`room_dim`) on the floorplan image. That shaped a 2-stage pipeline:

1. **Detection**: a `YOLOv8s` model, trained from scratch (via its `.yaml`
   architecture, using `ultralytics`) on an 80/20 train/val split, finds the
   `room_name` text boxes.
2. **Recognition**: each detected box is cropped and passed to EasyOCR,
   which reads the text (e.g. "KITCHEN", "BATH"). A substring-based
   classifier then buckets the text into `rooms` / `halls` / `kitchens` /
   `bathrooms` — see `src/floorplan.py:classify_room_label` and
   `tests/test_floorplan_classification.py` for its exact (imperfect)
   rules and known false positives.

**Metrics.** Two different things are worth measuring here, and they're not
the same number:

- **Detection quality (mAP@.5)**: how well YOLO finds the `room_name` boxes.
  The trained model reached **99.3%** mAP@.5 on the validation split.
- **End-to-end room-count accuracy**: whether the full YOLO+OCR+classifier
  pipeline gets the actual room counts right — a mAP score doesn't capture
  OCR or classification errors downstream of detection. This hasn't been
  measured rigorously against a held-out labeled set yet (the substring
  classifier's known false positives are documented and pinned by tests,
  not yet fixed or scored).

The training run originally produced two checkpoints (300 and 1000 epochs).
Both were benchmarked against a hand-labeled sample of `data/images/`; the
1000-epoch model had ~19% lower error and won 7 of 16 head-to-head
comparisons to the 300-epoch model's 2, so the 300-epoch checkpoint was
deleted rather than kept alongside it.

`notebooks/train.ipynb` has the full training/preprocessing code (COCO ->
YOLO format conversion, the train/val split, the training call).

---

## Development & Testing

```bash
pip install -r requirements.txt -r requirements-heavy.txt -r requirements-dev.txt

black --line-length 110 --check .
flake8 .

for f in tests/test_*.py; do python "$f"; done
# or, equivalently: pytest tests/
```

Tests use hand-written fakes and dependency injection (in-memory SQLite,
fake Qdrant client/embedder, `TestClient` with `app.dependency_overrides`)
rather than a live Postgres/Qdrant/LLM — see `tests/` for the pattern.
CI (`.github/workflows/ci.yml`) runs both lint and the full test suite on
every push to `main`.

---

## Model weights

`best_1000.pt` (~67MB) isn't committed to git. It's published as a
[GitHub release asset](https://github.com/AntiVicious/SmartSense-project/releases/tag/model-weights-v1),
and `src/Dockerfile` downloads it at build time (pinned to that release
tag and a SHA256 checksum, so a build never silently picks up a
different file).

- **Building with Docker**: nothing to do — `docker compose up --build`
  fetches it automatically.
- **Running outside Docker** (e.g. `notebooks/train.ipynb`, or calling
  `src/floorplan.py` directly): download it yourself and place it at
  `src/best_1000.pt`:
  ```bash
  curl -L -o src/best_1000.pt \
    https://github.com/AntiVicious/SmartSense-project/releases/download/model-weights-v1/best_1000.pt
  ```
- **Publishing a retrained model**: create a new release tag, upload the
  new `.pt` file as its asset, and update `MODEL_WEIGHTS_URL`/
  `MODEL_WEIGHTS_SHA256` in `src/Dockerfile` to match.
