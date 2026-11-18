# Cloud Deployment (retired)

SmartSense was deployed publicly on Google Cloud Run at $0/month using only
perpetual free tiers. The deployment has since been shut down; this note
records how it was set up.

## Architecture

Only `api` and `ui` are stateless, so only they ran on Cloud Run. The other
services were replaced or kept local:

| Piece | Service | Why |
|---|---|---|
| `api` and `ui` | Cloud Run, `--min-instances 0` | Always Free tier: 180,000 vCPU-seconds, 360,000 GiB-seconds and 2M requests per month. Services scale to zero between visits. |
| Postgres | [Neon](https://neon.tech) free tier | Cloud SQL has no free tier. Neon gives 0.5 GB and suspends when idle. |
| Vectors | [Qdrant Cloud](https://cloud.qdrant.io) free tier | 1 GB cluster; the dataset is a few MB. |
| Images | GitHub Container Registry | Free for public images. The ~3–4 GB compressed `api` image would exceed Artifact Registry's 0.5 GB free tier. |
| Builds | Local `docker build` and `docker push` | Avoids Cloud Build minutes for a multi-GB PyTorch image. |
| Secrets | Secret Manager | Five secrets, within the free tier. |
| Airflow | Not deployed | Airflow needs a long-running scheduler, which Cloud Run doesn't provide. The data was loaded once by running the local Airflow stack against Neon and Qdrant Cloud. |

GCP requires a billing account even for free-tier usage, so a $1 budget
alert was used as a safeguard. Cloud Run's free tier is a usage allowance,
not a hard cap: at 2 vCPU for `api`, it covers about 90,000 request-seconds
of processing per month.

## Configuration

`src/config.py` and `src/db.py` support managed services without code
changes:

- `POSTGRES_SSLMODE=require` for Neon, which rejects unencrypted
  connections.
- `QDRANT_HOST` accepts a full `https://` URL, with `QDRANT_API_KEY` for
  Qdrant Cloud.

The schema was created by running `alembic upgrade head` locally against
Neon.

## Deploy commands

Images were built locally and pushed to `ghcr.io`, with both packages set to
public so Cloud Run could pull them without credentials:

```bash
docker build -f src/Dockerfile.api -t ghcr.io/<user>/smartsense-api:latest .
docker build -f src/Dockerfile.ui -t ghcr.io/<user>/smartsense-ui:latest .
docker push ghcr.io/<user>/smartsense-api:latest
docker push ghcr.io/<user>/smartsense-ui:latest
```

`api` needs 4 GiB and 2 vCPU because it loads YOLO, EasyOCR and a
SentenceTransformer model:

```bash
gcloud run deploy smartsense-api \
  --image ghcr.io/<user>/smartsense-api:latest \
  --region us-central1 \
  --set-env-vars POSTGRES_HOST=<neon-host>,POSTGRES_SSLMODE=require,POSTGRES_USER=<user>,POSTGRES_DB=<db>,QDRANT_HOST=https://<cluster>.cloud.qdrant.io,INTERNAL_API_KEY=<random> \
  --set-secrets POSTGRES_PASSWORD=postgres-password:latest,GROQ_API_KEY=groq-api-key:latest,API_KEY=api-key:latest,QDRANT_API_KEY=qdrant-api-key:latest \
  --memory 4Gi --cpu 2 \
  --min-instances 0 --max-instances 3 \
  --timeout 300 \
  --allow-unauthenticated

gcloud run deploy smartsense-ui \
  --image ghcr.io/<user>/smartsense-ui:latest \
  --region us-central1 \
  --set-env-vars API_BASE_URL=<smartsense-api-url> \
  --set-secrets API_KEY=api-key:latest \
  --memory 512Mi --cpu 1 \
  --min-instances 0 --max-instances 3 \
  --allow-unauthenticated
```

## Trade-offs

- **Cold starts.** With no minimum instances, the first request after an idle
  period waits for `api` to load its models.
- **No live ingestion.** Without Airflow in the cloud, the public deployment
  served a pre-loaded dataset; ingestion ran locally.
- **LLM cost outside GCP.** `/chat` is unauthenticated and calls Groq for each
  message, so usage is bounded by Groq's free-tier limits rather than GCP's.
