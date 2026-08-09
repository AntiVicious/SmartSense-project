"""The property ingestion pipeline: Excel -> Postgres + Qdrant.

ingest_properties_sync() is synchronous and CPU/IO-heavy (pandas, YOLO,
EasyOCR, DB writes) — it's meant to be run via run_in_threadpool from the
async /ingest route, not awaited directly.
"""

import io
import os

import numpy as np
import pandas as pd
from fastapi import HTTPException
from qdrant_client.http.models import PointStruct

from .documents import parse_local_pdf
from .floorplan import parse_floorplan
from .models import Property


def ingest_properties_sync(
    file_contents: bytes,
    *,
    session_factory,
    qdrant_client,
    embedder,
    qdrant_collection: str,
    parse_floorplan_fn=parse_floorplan,
    parse_local_pdf_fn=parse_local_pdf,
) -> dict:
    """parse_floorplan_fn/parse_local_pdf_fn default to the real YOLO/OCR
    and PyMuPDF-backed parsers -- production callers never need to pass
    them. Tests inject fakes here instead of loading real models."""
    db = session_factory()
    try:
        df = pd.read_excel(io.BytesIO(file_contents))

        # --- Data Cleaning ---
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.replace({np.nan: None})
        # -----------------------

        print(f"Read {len(df)} rows from Excel. Cleaned price data.")

        qdrant_points = []
        ingested = 0

        for index, row in df.iterrows():
            # --- Use .get() for safety ---
            image_filename = row.get('image_file')
            certs_link_str = str(row.get('certificates', '')) # Get certs as string
            long_desc = row.get('long_description')
            price_val = row.get('price')
            title_val = row.get('title')
            location_val = row.get('location')

            if not image_filename:
                print(f"Skipping row {index}: No image filename.")
                continue

            # Construct the local path to the image
            local_image_path = os.path.join("/app/data/images", str(image_filename))

            floorplan_data = parse_floorplan_fn(local_image_path)
            if floorplan_data.get("error"):
                print(f"Skipping row {index}: {floorplan_data['error']}")
                continue

            # --- NEW PDF PARSING LOGIC ---
            report_text = ""
            if certs_link_str:
                links = certs_link_str.split('|') # Split by pipe
                for link in links:
                    if link and link.strip().lower().endswith('.pdf'):
                        print(f"Found PDF link: {link.strip()}")
                        pdf_path = os.path.join("/app/data/certificates", link.strip())
                        report_text += parse_local_pdf_fn(pdf_path) + "\n\n"
            # ---------------------------------

            db_property = Property(
                title=title_val,
                description=long_desc,
                location=location_val,
                price=price_val,
                listing_date=row.get('listing_date'),
                certifications_link=certs_link_str,
                floorplan_image_url=image_filename,
                rooms=floorplan_data.get('rooms'),
                halls=floorplan_data.get('halls'),
                kitchens=floorplan_data.get('kitchens'),
                bathrooms=floorplan_data.get('bathrooms')
            )
            db.add(db_property)
            db.flush()  # populate db_property.id without committing yet
            sql_id = db_property.id

            # --- UPDATED TEXT FOR EMBEDDING ---
            text_to_embed = f"Title: {title_val}. Description: {long_desc}. Location: {location_val}. Reports: {report_text}"
            embedding = embedder.embed_query(text_to_embed)

            payload = {"text": text_to_embed, "property_id": sql_id}
            qdrant_points.append(PointStruct(id=sql_id, vector=embedding, payload=payload))
            ingested += 1

        db.commit()

        if qdrant_points:
            qdrant_client.upsert(
                collection_name=qdrant_collection,
                points=qdrant_points,
                wait=True
            )

        return {"status": "success", "message": f"Successfully ingested {ingested} properties."}

    except KeyError as e:
        db.rollback()
        print(f"Ingestion error: Missing column {e}")
        raise HTTPException(status_code=400, detail=f"Missing column in Excel file: {e}")
    except Exception as e:
        db.rollback()
        import traceback
        traceback.print_exc()
        print(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion Error: {str(e)}")
    finally:
        db.close()
