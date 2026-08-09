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


def prepare_ocr_crop(img_pil: Image.Image, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Pads and upscales a detected room_name box before handing it to OCR.

    eval/REPORT.md's held-out measurement of classify_room_label found
    that *every* one of its real-category misclassifications traced back
    to OCR returning empty or corrupted text, not to the substring rules
    -- and that this tracked crop size: boxes OCR failed on averaged
    28x14px, versus 57x17px for boxes it read successfully, and crops
    that failed were confirmed by eye to contain perfectly legible text
    (e.g. "DINING", "BATH") at 5x upscale. This is that fix: pad the box
    by 15% a side (a tight YOLO box can clip a character's edge) and
    upscale so the crop is at least MIN_HEIGHT tall, capped at MAX_SCALE
    to avoid blowing up already-adequate crops into blur.
    """
    MIN_HEIGHT = 48
    MAX_SCALE = 6

    img_w, img_h = img_pil.size
    box_w, box_h = x2 - x1, y2 - y1
    pad_x = max(2, int(box_w * 0.15))
    pad_y = max(2, int(box_h * 0.15))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_w, x2 + pad_x)
    y2 = min(img_h, y2 + pad_y)

    crop = img_pil.crop((x1, y1, x2, y2))

    scale = min(MAX_SCALE, max(1.0, MIN_HEIGHT / max(1, crop.height)))
    if scale > 1.0:
        new_size = (max(1, int(crop.width * scale)), max(1, int(crop.height * scale)))
        crop = crop.resize(new_size, Image.LANCZOS)

    return np.array(crop)


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

                cropped_img_np = prepare_ocr_crop(img_pil, x1, y1, x2, y2)

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
