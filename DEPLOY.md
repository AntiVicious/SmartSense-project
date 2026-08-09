# Deploying to Google Cloud Run

This is a plan, not a runbook that's been executed — nothing here has been
provisioned. Read the [Architecture decision](#architecture-decision)
section before running anything; it's the one call that changes both the
steps and the cost below.

## Why this isn't a plain `gcloud run deploy`

`api` and `ui` are the only two services in this stack that are actually
stateless. Everything else has a reason it isn't a Cloud Run candidate
as-is:

- **Postgres / Qdrant** need a real, persistent disk. Cloud Run instances
  don't keep local disk across restarts or scale-to-zero — this is exactly
  why they're managed services below, not containers we deploy.
- **Airflow** (scheduler + webserver) needs a long-running process — Cloud
  Run kills idle instances, which is incompatible with a scheduler that has
  to keep polling. Worse: `watch_ingest_landing_dag.py`'s trigger is a
  shared POSIX directory (`INGEST_LANDING_DIR`) that `api` and Airflow both
  read/write via a plain bind mount in `docker-compose.yml`. Cloud Run
  services don't share a local filesystem with each other or with a VM —
  that mechanism doesn't survive the move unchanged.

That last point is the real architectural decision, not a detail to skip
past.

## Architecture decision

Three options, cheapest/simplest first. **I'd deploy Tier A** — it's the
one that matches what this deployment is actually for (a live link people
can click), at the lowest cost and least new surface area. Say the word if
you want B or C instead; the steps and cost below are written for A, and
I'll redo them if you pick differently.

### Tier A — Demo-only (recommended)

Deploy `api` + `ui` to Cloud Run. Postgres → Cloud SQL. Qdrant → Qdrant
Cloud's free tier. **Airflow stays local** — not deployed anywhere; it
keeps doing exactly what it's doing right now in `docker-compose.yml`.

The public deployment's job is to let people chat and search against a
dataset that's already there, not to run your ingestion pipeline for
strangers. Populate Cloud SQL + Qdrant Cloud once, before or at deploy
time, by pointing your local Airflow stack at the cloud databases for a
single backfill run (swap `POSTGRES_HOST`/`QDRANT_HOST` in `.env` to the
Cloud SQL/Qdrant Cloud endpoints, run `scripts/backfill_ingest.py` locally,
switch back). `POST /ingest` still works in the deployed API — it enqueues
a job and returns a `job_id` — but with no Airflow watching the cloud
deployment, nothing ever picks it up, so the job sits at `queued` forever.
That's a real rough edge, not a crash: **disable the "Start Ingestion"
button in the deployed `ui`** (an env var like `SHOW_INGEST_UI=false`
gating that `st.header("Data Ingestion")` block, or just don't wire it up
for this deploy) so it doesn't look broken to someone testing it live.

Cost driver: Cloud SQL, running 24/7 because it doesn't scale to zero the
way Cloud Run does. Everything else is at or near GCP's free tier for
demo-level traffic.

### Tier B — Full pipeline, cost-optimized

Everything in A, plus: Airflow (webserver + scheduler + its own metadata
Postgres) on a small always-on Compute Engine VM — essentially today's
`docker-compose.yml` minus `api`/`ui`/`postgres-db`/`qdrant-db`, running on
a persistent VM instead of your laptop. Bridge the file-landing trigger
with a GCS bucket mounted via `gcsfuse` on both the VM and Cloud Run's
`api` service (Cloud Run supports GCS FUSE volume mounts natively now) —
`INGEST_LANDING_DIR` points at the mount, and `os.rename`/`os.listdir` in
`ingest_landing.py` and `watch_ingest_landing_dag.py` keep working mostly
unchanged. Flagging honestly: gcsfuse's rename semantics aren't identical
to a real POSIX filesystem's atomic rename — worth a real concurrency test
before trusting it the way the current bind-mount is trusted, not
something I'd assume works identically without checking.

Cost driver: the VM, running 24/7, on top of Tier A's Cloud SQL.

### Tier C — Cloud Composer (not recommended for this)

Replace self-hosted Airflow with GCP's fully managed Airflow. The "proper"
managed-service answer, and worth knowing about — but Composer 2's
smallest environment has a fixed floor around $300+/month (it provisions a
GKE cluster, Cloud SQL, and Redis underneath, all always-on) regardless of
how little you use it. Disproportionate to a 73-row portfolio dataset.
Mentioning it so you know it was considered and why it's not the pick.

---

## Deployment steps (Tier A)

### 0. Prerequisites

- A GCP project with billing enabled
- `gcloud` CLI authenticated (`gcloud auth login`, `gcloud config set project <PROJECT_ID>`)
- APIs enabled: `run.googleapis.com`, `sqladmin.googleapis.com`,
  `artifactregistry.googleapis.com`, `secretmanager.googleapis.com`
- A [Qdrant Cloud](https://cloud.qdrant.io/) account (free tier, 1GB cluster)
- Groq API key (already have this for local dev)

### 1. Cloud SQL for PostgreSQL

```bash
gcloud sql instances create smartsense-db \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --region=us-central1 \
  --storage-size=10GB \
  --storage-type=SSD

gcloud sql databases create real_estate_db --instance=smartsense-db
gcloud sql users create smartsense_app --instance=smartsense-db --password=<generate-one>
```

Note the instance's **connection name**
(`PROJECT_ID:us-central1:smartsense-db`) — that becomes
`POSTGRES_HOST=/cloudsql/PROJECT_ID:us-central1:smartsense-db`
(`src/config.py`'s `database_url` already detects and handles this
Unix-socket form; Cloud Run mounts it automatically when you pass
`--add-cloudsql-instances` at deploy time, no Cloud SQL Auth Proxy sidecar
needed).

### 2. Qdrant Cloud

Create a free-tier cluster at cloud.qdrant.io. Note its URL (the full
`https://...` URL) and API key — set `QDRANT_HOST=<that URL>` and
`QDRANT_API_KEY=<key>` (`src/db.py`'s `get_qdrant_client()` already
detects a URL vs. a bare hostname and passes the API key either way).

### 3. Secrets

```bash
printf '%s' '<groq-key>' | gcloud secrets create groq-api-key --data-file=-
printf '%s' '<generated>' | gcloud secrets create api-key --data-file=-
printf '%s' '<qdrant-api-key>' | gcloud secrets create qdrant-api-key --data-file=-
printf '%s' '<db-password>' | gcloud secrets create postgres-password --data-file=-
```

(`INTERNAL_API_KEY` isn't needed in this tier — nothing calls
`/internal/*` without Airflow running against this deployment.)

### 4. Build and push images

```bash
gcloud artifacts repositories create smartsense --repository-format=docker --location=us-central1

gcloud builds submit --tag us-central1-docker.pkg.dev/PROJECT_ID/smartsense/api -f src/Dockerfile.api .
gcloud builds submit --tag us-central1-docker.pkg.dev/PROJECT_ID/smartsense/ui -f src/Dockerfile.ui .
```

### 5. Deploy `api`

```bash
gcloud run deploy smartsense-api \
  --image us-central1-docker.pkg.dev/PROJECT_ID/smartsense/api \
  --region us-central1 \
  --add-cloudsql-instances PROJECT_ID:us-central1:smartsense-db \
  --set-env-vars POSTGRES_HOST=/cloudsql/PROJECT_ID:us-central1:smartsense-db,POSTGRES_USER=smartsense_app,POSTGRES_DB=real_estate_db,QDRANT_HOST=https://<your-cluster>.cloud.qdrant.io,LOG_LEVEL=INFO \
  --set-secrets POSTGRES_PASSWORD=postgres-password:latest,GROQ_API_KEY=groq-api-key:latest,API_KEY=api-key:latest,INTERNAL_API_KEY=api-key:latest,QDRANT_API_KEY=qdrant-api-key:latest \
  --memory 4Gi --cpu 2 \
  --min-instances 0 --max-instances 3 \
  --timeout 300 \
  --no-allow-unauthenticated=false
```

`--min-instances 0`: scales to zero between visits — the cost decision
below. `--memory 4Gi --cpu 2`: this image loads YOLO + EasyOCR +
SentenceTransformer; test against something smaller first if you want to
trim cost, but don't assume it'll fit in Cloud Run's default 512Mi without
checking. `INTERNAL_API_KEY` reuses `API_KEY`'s secret here since nothing
calls `/internal/*` in this tier anyway — harmless, not meaningful.

Run the Alembic migration once against Cloud SQL before first use (same
pattern as local: `alembic upgrade head`, pointed at the Cloud SQL
instance via the Cloud SQL Auth Proxy run locally, or `gcloud sql connect`).

### 6. Deploy `ui`

```bash
gcloud run deploy smartsense-ui \
  --image us-central1-docker.pkg.dev/PROJECT_ID/smartsense/ui \
  --region us-central1 \
  --set-env-vars API_BASE_URL=<smartsense-api-url-from-step-5> \
  --set-secrets API_KEY=api-key:latest \
  --memory 512Mi --cpu 1 \
  --min-instances 0 --max-instances 3 \
  --allow-unauthenticated
```

`ui` is the one that should be `--allow-unauthenticated` — it's the public
demo page. `api` can be locked to authenticated invocations if you want
`ui` as the only caller (Cloud Run service-to-service IAM), or left open
behind its own `X-API-Key` check on the write paths — your call on how far
to take that; the app-level auth from the Hardening section holds either
way.

### 7. Populate the data once

```bash
# Point local Airflow at the cloud databases for one backfill run --
# swap POSTGRES_HOST/QDRANT_HOST in .env to the Cloud SQL/Qdrant Cloud
# endpoints (Cloud SQL needs the Auth Proxy running locally for this),
# then:
docker compose run --rm api python scripts/backfill_ingest.py data/backfill/
# ... let the local Airflow stack process it against the cloud DBs ...
# then switch .env back to the local docker-compose values.
```

---

## Cost estimate (Tier A, `us-central1`)

**These are ballpark figures from GCP's public list pricing, not a quote —
run this through the
[GCP Pricing Calculator](https://cloud.google.com/products/calculator)
with your actual region and usage before trusting them for a decision.**
Cloud Run's free tier (per month, doesn't expire): 180,000 vCPU-seconds,
360,000 GiB-seconds, 2M requests.

| Component | Assumption | Est. monthly cost |
|---|---|---|
| Cloud Run `api` | min-instances=0, light demo traffic (well under free tier) | **$0–5** |
| Cloud Run `ui` | min-instances=0, light demo traffic | **$0–2** |
| Cloud SQL (`db-f1-micro`, 10GB SSD) | 24/7 — Cloud SQL doesn't scale to zero | **$10–15** |
| Qdrant Cloud | free tier (1GB cluster, this dataset is ~73 rows) | **$0** |
| Artifact Registry | a few images, a few GB | **~$0.50** |
| Egress | low-traffic demo | **~$0–1** |
| **Total** | | **~$12–25/month** |

**Cold starts are the real tradeoff for min-instances=0, not just cost.**
`api`'s lifespan loads YOLO + EasyOCR + SentenceTransformer and builds the
LangChain agent before `/health` answers — 20–40s locally, plausibly
longer on a cold Cloud Run instance. `main.py` already handles this
gracefully (shows "Backend is starting up…" and polls), so a cold demo
isn't broken, just slow on the first hit after idle. If that's not
acceptable for a recruiter's first impression, `--min-instances 1` removes
it — at a real, continuous cost: roughly 2 vCPU + 4GiB running 24/7 prices
out to **~$120–150/month** by itself, which is why it's not the default
recommendation here.

**Not counted above, and separate from GCP billing:** Groq API usage
(pay-per-token; check their current pricing and consider a spend cap —
`/chat` is the one endpoint intentionally left open to the public in this
deployment, so it's the one to rate-limit or budget-alert on if the demo
gets real traffic).

**Cutting cost further:** `gcloud sql instances patch smartsense-db
--activation-policy=NEVER` stops Cloud SQL between demos (billed for
storage only while stopped, ~$1–2/month) — worth doing if this is a
"share the link when someone asks" project rather than an always-on demo.

---

## What I'd want confirmed before running any of this

1. **Tier A, B, or C** — the architecture section above.
2. Whether `/chat` staying open to the public (no `API_KEY` gate — that's
   deliberate, see the Hardening section in `README.md`) is acceptable
   for however long this stays deployed, given it's the one endpoint that
   costs real money (Groq) per use.
3. GCP project / billing account to deploy into, and preferred region.

Nothing above has been run.
