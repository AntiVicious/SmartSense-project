"""PDF text extraction: local certificate files and remote URLs."""

import os

import fitz
import requests


def parse_local_pdf(local_pdf_path: str) -> str:
    """Opens a local PDF file and extracts all text."""
    if not local_pdf_path or not os.path.exists(local_pdf_path):
        print(f"Warning: PDF file not found at {local_pdf_path}")
        return ""

    try:
        pdf_text = ""
        with fitz.open(local_pdf_path) as doc:
            for page in doc:
                pdf_text += page.get_text()

        print(f"Successfully parsed {len(pdf_text)} chars from {os.path.basename(local_pdf_path)}")
        return pdf_text

    except Exception as e:
        print(f"Warning: Could not parse PDF {local_pdf_path}. Error: {e}")
        return "" # Return empty string on failure


def fetch_and_parse_pdf(pdf_url: str) -> str:
    """Downloads a PDF from a URL and extracts all text."""
    if not pdf_url or not isinstance(pdf_url, str) or not pdf_url.lower().endswith('.pdf'):
        return "" # Return empty string if no valid PDF URL

    try:
        print(f"Fetching PDF from: {pdf_url}")
        response = requests.get(pdf_url, timeout=10) # 10 sec timeout
        response.raise_for_status() # Raise error if bad response (404, 500)

        pdf_text = ""
        # Open the PDF from in-memory bytes
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            for page in doc:
                pdf_text += page.get_text()

        print(f"Successfully parsed {len(pdf_text)} chars from {pdf_url}")
        return pdf_text

    except Exception as e:
        print(f"Warning: Could not parse PDF from {pdf_url}. Error: {e}")
        return "" # Return empty string on failure
