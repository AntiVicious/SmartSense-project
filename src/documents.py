"""PDF text extraction: local certificate files and remote URLs.

Currently unreferenced: dags/ingest_properties_dag.py's
extract_certificate_text task does its own inline PyMuPDF extraction
rather than calling either function here (a leftover from before the
Airflow migration). Flagging rather than deleting -- out of scope for
the change that found it."""

import logging
import os

import fitz
import requests

logger = logging.getLogger(__name__)


def parse_local_pdf(local_pdf_path: str) -> str:
    """Opens a local PDF file and extracts all text."""
    if not local_pdf_path or not os.path.exists(local_pdf_path):
        logger.warning("PDF file not found at %s", local_pdf_path)
        return ""

    try:
        pdf_text = ""
        with fitz.open(local_pdf_path) as doc:
            for page in doc:
                pdf_text += page.get_text()

        logger.debug("Parsed %d chars from %s", len(pdf_text), os.path.basename(local_pdf_path))
        return pdf_text

    except Exception:
        logger.warning("Could not parse PDF %s", local_pdf_path, exc_info=True)
        return ""  # Return empty string on failure


def fetch_and_parse_pdf(pdf_url: str) -> str:
    """Downloads a PDF from a URL and extracts all text."""
    if not pdf_url or not isinstance(pdf_url, str) or not pdf_url.lower().endswith(".pdf"):
        return ""  # Return empty string if no valid PDF URL

    try:
        logger.debug("Fetching PDF from: %s", pdf_url)
        response = requests.get(pdf_url, timeout=10)  # 10 sec timeout
        response.raise_for_status()  # Raise error if bad response (404, 500)

        pdf_text = ""
        # Open the PDF from in-memory bytes
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            for page in doc:
                pdf_text += page.get_text()

        logger.debug("Parsed %d chars from %s", len(pdf_text), pdf_url)
        return pdf_text

    except Exception:
        logger.warning("Could not parse PDF from %s", pdf_url, exc_info=True)
        return ""  # Return empty string on failure
