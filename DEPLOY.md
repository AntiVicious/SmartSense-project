# Deploying to Google Cloud Run — $0/month

This is a plan, not a runbook that's been executed — nothing here has been
provisioned. Every piece below is chosen specifically to stay inside a
genuine, perpetual free tier — not a 90-day trial credit, not "should be
cheap." Where something can't be made free, it's called out explicitly
rather than glossed over.

## Why not just `gcloud run deploy` everything

`api` and `ui` are the only two services in this stack that are actually
stateless — those go on Cloud Run. Everything else needs a real answer:

- **Postgres** needs persistent storage Cloud Run doesn't provide, and
  **Cloud SQL has no free tier at all** — it bills 24/7 the moment it
  exists, free-tier project or not. Swapped for **Neon**, a serverless
  Postgres host with a genuinely free tier that (like Cloud Run) suspends
  to zero when idle — the same shape as everything else here.
- **Qdrant** likewise needs persistent storage. **Qdrant Cloud**'s free
  tier (1GB cluster) comfortably covers this project's 73-row dataset.
- **Airflow** needs a long-running scheduler process, which Cloud Run
  doesn't provide (it kills idle instances), and its file-landing trigger
  (`watch_ingest_landing_dag.py`) depends on a POSIX directory shared with
  `api` via a plain bind mount today — that mechanism doesn't survive
  `api` moving to Cloud Run unchanged. The only genuinely free way to keep
  Airflow *running* is Compute Engine's single free `e2-micro` instance —
  1 shared vCPU, 1GB RAM, which is tight for webserver + scheduler + its
  own Postgres together and **I haven't verified it fits**. Rather than
  guess, this plan keeps Airflow **local, not deployed** — the same
  container stack you already have running, on your machine.

The public deployment's job, then, is to let people chat and search a
dataset that's already there — not to run ingestion for strangers.
Populate Neon + Qdrant Cloud once via a local backfill run; disable the
"Start Ingestion" button in the deployed `ui` (an env var like
`SHOW_INGEST_UI=false` gating that block in `main.py`, not yet wired up —
say the word and I'll add it) so `POST /ingest` enqueueing into a queue
nothing ever drains doesn't look broken to someone testing it live.

If you want the full pipeline live in the cloud later, that's the
`e2-micro`-fits-Airflow question to actually test first — worth doing as
its own follow-up, not folded into "make it free."

## What's free, and why

| Piece | Service | Why it's $0 |
|---|---|---|
| `api` + `ui` compute | Cloud Run | "Always Free" tier (perpetual, not a trial): 180,000 vCPU-sec + 360,000 GiB-sec + 2M requests/month, with `--min-instances 0`. A low-traffic demo doesn't come close — see the math below. |
| Postgres | [Neon](https://neon.tech) free tier | 0.5GB storage, autosuspends when idle — same "scales to zero" shape as Cloud Run, not a 24/7-billed instance like Cloud SQL. |
| Vectors | [Qdrant Cloud](https://cloud.qdrant.io) free tier | 1GB cluster, this dataset is a few MB. |
| Container images | GitHub Container Registry (`ghcr.io`) | Free, unlimited storage for public images. Artifact Registry's free tier is only 0.5GB — `api`'s image (~3-4GB compressed) would exceed it and start billing ~$0.10/GB/month; skipped entirely by using `ghcr.io` instead. |
| Image builds | Local `docker build` + `docker push` | Cloud Build has a daily free-minutes tier, but building a multi-GB image (PyTorch, EasyOCR) could plausibly exceed it. Building locally (same as every image you've already built this whole engagement) uses zero Cloud Build minutes. |
| Secrets | Secret Manager | Free tier covers 6 active secret versions/month; this deployment needs ~5 (`GROQ_API_KEY`, `API_KEY`, `INTERNAL_API_KEY`, Neon password, Qdrant API key). |
| Airflow | Not deployed | Stays on your machine. $0 because nothing runs in the cloud. |

**One unavoidable non-technical requirement:** GCP requires a billing
account (a payment method on file) to enable Cloud Run at all, even for
$0 usage — this is a GCP account-verification step, not a charge. Set a
[Billing Budget alert](https://console.cloud.google.com/billing/budgets)
at, say, $1 as a tripwire — free, takes two minutes, and emails you the
moment anything would cost money instead of finding out on a bill.

**The one real boundary worth understanding, not just trusting:** Cloud
Run's free tier is usage-based, not a hard cap that blocks requests —
if traffic is ever high enough to exceed it, GCP bills the overage rather
than stopping the service. For this project's `api` (2 vCPU / 4GiB, since
it's loading YOLO + EasyOCR + SentenceTransformer) that ceiling is
180,000 ÷ 2 = **90,000 request-seconds of actual processing per month**
before anything is billed — for a portfolio demo, that's hundreds of
requests a day, every day, sustained. The budget alert above is the real
safety net if that assumption ever turns out wrong.

---

## Deployment steps

### 0. Prerequisites

- A GCP project with billing *enabled* (see above — free tier still
  needs this)
- `gcloud` CLI authenticated (`gcloud auth login`, `gcloud config set project <PROJECT_ID>`)
- APIs enabled: `run.googleapis.com`, `secretmanager.googleapis.com`
- A [Neon](https://neon.tech) account (free tier)
- A [Qdrant Cloud](https://cloud.qdrant.io) account (free tier)
- A GitHub account with a
  [personal access token](https://github.com/settings/tokens) scoped to
  `write:packages` (to push to `ghcr.io`) — the repo is already public,
  so pulling the image back down at deploy time needs no auth at all
- Groq API key (already have this for local dev)

### 1. Neon Postgres

Create a free project at neon.tech. It gives you a full connection
string; pull the pieces out of it for `.env`:

```bash
POSTGRES_HOST=ep-xxxx-xxxx.us-east-2.aws.neon.tech
POSTGRES_USER=<from Neon>
POSTGRES_PASSWORD=<from Neon>
POSTGRES_DB=<from Neon>
POSTGRES_SSLMODE=require   # Neon rejects unencrypted connections -- src/config.py now supports this
```

Run the migration once, from your machine, against Neon directly (no
proxy needed — Neon is a normal public TCP endpoint):

```bash
POSTGRES_HOST=ep-xxxx... POSTGRES_SSLMODE=require alembic upgrade head
```

### 2. Qdrant Cloud

Create a free-tier cluster at cloud.qdrant.io. Note its full URL
(`https://xyz.cloud.qdrant.io`) and API key —
`QDRANT_HOST=<that URL>`, `QDRANT_API_KEY=<key>`
(`src/db.py`'s `get_qdrant_client()` already detects a URL vs. a bare
hostname and passes the key either way — no code change needed here).

### 3. Secrets

```bash
printf '%s' '<groq-key>' | gcloud secrets create groq-api-key --data-file=-
printf '%s' '<generated>' | gcloud secrets create api-key --data-file=-
printf '%s' '<neon-password>' | gcloud secrets create postgres-password --data-file=-
printf '%s' '<qdrant-api-key>' | gcloud secrets create qdrant-api-key --data-file=-
```

(`INTERNAL_API_KEY` isn't meaningfully used in this deployment — nothing
calls `/internal/*` without Airflow running against it. Set it to any
random value so `Settings` doesn't fail to start; it just won't be
exercised.)

### 4. Build and push images (no Cloud Build)

```bash
echo "<github-pat>" | docker login ghcr.io -u <your-github-username> --password-stdin

docker build -f src/Dockerfile.api -t ghcr.io/<you>/smartsense-api:latest .
docker build -f src/Dockerfile.ui -t ghcr.io/<you>/smartsense-ui:latest .

docker push ghcr.io/<you>/smartsense-api:latest
docker push ghcr.io/<you>/smartsense-ui:latest
```

Then make both packages public on GitHub (Package settings → Change
visibility) so Cloud Run can pull them without credentials.

### 5. Deploy `api`

```bash
gcloud run deploy smartsense-api \
  --image ghcr.io/<you>/smartsense-api:latest \
  --region us-central1 \
  --set-env-vars POSTGRES_HOST=ep-xxxx-xxxx.us-east-2.aws.neon.tech,POSTGRES_SSLMODE=require,POSTGRES_USER=<neon-user>,POSTGRES_DB=<neon-db>,QDRANT_HOST=https://<your-cluster>.cloud.qdrant.io,LOG_LEVEL=INFO \
  --set-secrets POSTGRES_PASSWORD=postgres-password:latest,GROQ_API_KEY=groq-api-key:latest,API_KEY=api-key:latest,QDRANT_API_KEY=qdrant-api-key:latest \
  --set-env-vars INTERNAL_API_KEY=unused-placeholder \
  --memory 4Gi --cpu 2 \
  --min-instances 0 --max-instances 3 \
  --timeout 300 \
  --allow-unauthenticated
```

`--min-instances 0`: this is the whole cost story — scales to zero
between visits. `--memory 4Gi --cpu 2`: this image loads YOLO + EasyOCR +
SentenceTransformer; test against something smaller first if you want to
trim the (already-$0) footprint, but don't assume it fits Cloud Run's
default 512Mi without checking.

### 6. Deploy `ui`

```bash
gcloud run deploy smartsense-ui \
  --image ghcr.io/<you>/smartsense-ui:latest \
  --region us-central1 \
  --set-env-vars API_BASE_URL=<smartsense-api-url-from-step-5> \
  --set-secrets API_KEY=api-key:latest \
  --memory 512Mi --cpu 1 \
  --min-instances 0 --max-instances 3 \
  --allow-unauthenticated
```

### 7. Populate the data once

```bash
# Point local Airflow at the cloud databases for one backfill run --
# swap POSTGRES_HOST/POSTGRES_SSLMODE/QDRANT_HOST in .env to Neon/Qdrant
# Cloud, run the existing local stack against them:
docker compose run --rm api python scripts/backfill_ingest.py data/backfill/
# ... let the local Airflow stack process it against the cloud DBs ...
# then switch .env back to the local docker-compose values.
```

---

## What I'd want confirmed before running any of this

1. This is now a single recommended path (no more tiered options) since
   the other tiers this doc previously described weren't actually free —
   say so if you want the full-pipeline-in-the-cloud version explored
   instead, with the `e2-micro`-fits-Airflow question tested first.
2. GCP project / billing account to deploy into (billing enabled, budget
   alert set), and preferred region.
3. Whether `/chat` staying open to the public (no `API_KEY` gate — see
   the Hardening section in `README.md`) is fine given it calls Groq's
   API per message — that's a cost surface outside GCP entirely; check
   Groq's current free-tier limits before this goes live.

Nothing above has been run.
