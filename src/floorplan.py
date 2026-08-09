"""Floorplan parsing: YOLO room-box detection + EasyOCR text classification.

The YOLO and EasyOCR models are lazy-loaded module-level singletons, same
as before the split — the first call to parse_floorplan() loads them, every
call after that reuses them. Nothing here touches disk or a model at
import time.
"""

import os
import re
from pathlib import Path

import easyocr
import numpy as np
from PIL import Image
from ultralytics import YOLO

FLOORPLAN_MODEL_PATH = str(Path(__file__).resolve().parent / "best_1000.pt")

_yolo_model = None  # Lazy-load the YOLO model
_ocr_reader = None  # Lazy-load the OCR model


def classify_room_label(detected_text: str) -> str:
    """Classify a cleaned OCR label into one of the room-count buckets.

    detected_text must already be lowercased and stripped down to letters
    and whitespace (see the re.sub call in parse_floorplan below) -- this
    function does no cleaning itself, matching how the classification was
    always invoked inline before it was pulled out as its own function.

    Branch order is significant, not just style: "bathroom" contains both
    "bath" and "room", and only resolves to "bathrooms" because the
    bathroom check runs before the room check. Several of these substring
    checks also false-positive on unrelated words -- "ki" matches
    "parking", "br" matches "library" and "breakfast" -- see the tests in
    tests/test_floorplan_classification.py, which document the current
    (imperfect) behavior rather than silently rely on it.
    """
    if "ki" in detected_text:
        return "kitchens"
    elif (
        "bath" in detected_text
        or "wc" in detected_text
        or "wash" in detected_text
        or "toi" in detected_text
        or "powder" in detected_text
    ):
        return "bathrooms"
    elif "hall" in detected_text or "liv" in detected_text or "great" in detected_text:
        return "halls"
    elif "bed" in detected_text or "room" in detected_text or "br" in detected_text:
        # This is a general "room", e.g., bedroom
        return "rooms"
    else:
        return "others"


def parse_floorplan(local_image_path: str) -> dict:
    global _yolo_model, _ocr_reader

    # Lazy-load YOLO model
    if _yolo_model is None:
        if not os.path.exists(FLOORPLAN_MODEL_PATH):
            print(f"Error: Model file not found at {FLOORPLAN_MODEL_PATH}")
            return {"error": f"Model file not found: {FLOORPLAN_MODEL_PATH}"}
        print(f"Loading floorplan model {FLOORPLAN_MODEL_PATH}...")
        _yolo_model = YOLO(FLOORPLAN_MODEL_PATH)

    model = _yolo_model

    # Lazy-load OCR model
    if _ocr_reader is None:
        print("Lazy loading OCR model (EasyOCR)...")
        _ocr_reader = easyocr.Reader(["en"])
        print("OCR model loaded.")

    if not os.path.exists(local_image_path):
        print(f"Error: Image file not found at {local_image_path}")
        return {"error": f"Image file not found: {local_image_path}"}

    print(f"Parsing image: {local_image_path}")

    try:
        img_pil = Image.open(local_image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {local_image_path}: {e}")
        return {"error": f"Could not open image: {e}"}

    # Run YOLO detection
    results = model.predict(img_pil, imgsz=640, conf=0.25)
    result = results[0]

    counts = {"rooms": 0, "halls": 0, "kitchens": 0, "bathrooms": 0, "others": 0}
    class_names = result.names

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = class_names[class_id]

            if label == "room_name":
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords

                cropped_img_pil = img_pil.crop((x1, y1, x2, y2))
                cropped_img_np = np.array(cropped_img_pil)

                ocr_result_list = _ocr_reader.readtext(cropped_img_np, detail=0)

                if ocr_result_list:
                    detected_text = " ".join(ocr_result_list).lower()
                    detected_text = re.sub(r"[^a-z\s]", "", detected_text).strip()
                    print(f"  > Detected label '{label}' -> OCR Text: '{detected_text}'")

                    counts[classify_room_label(detected_text)] += 1

    json_output = {
        "rooms": counts["rooms"],
        "halls": counts["halls"],
        "kitchens": counts["kitchens"],
        "bathrooms": counts["bathrooms"],
        "other rooms": counts["others"],
    }

    return json_output
