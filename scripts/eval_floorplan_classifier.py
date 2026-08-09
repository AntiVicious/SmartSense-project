#!/usr/bin/env python
"""Measures src/floorplan.py's classify_room_label() against real detections
on a held-out split of data/images/ -- Tier 4 in smartsense-fix-list.md.

Two subcommands, run in sequence:

  extract   Splits the image pool into a held-out eval set and a dev
            remainder (fixed seed, so the split is reproducible), runs the
            *real* YOLO + EasyOCR pipeline over the eval set only, and
            writes one row per detected room_name box to a CSV -- OCR
            text, this repo's actual classify_room_label() prediction,
            and an empty ground_truth_class column for a human to fill in
            by looking at the saved crop.

  score     Reads that CSV back (after ground_truth_class has been filled
            in by hand) and computes the confusion matrix and per-class
            precision/recall, each number next to its support count.

Deliberately calls the real classify_room_label() from src.floorplan
rather than reimplementing the substring rules here -- this measures the
shipped classifier, not a description of it. YOLO detection and OCR are
also the real pipeline (same imgsz/conf/cleaning as parse_floorplan), so
the eval reflects what the deployed pipeline actually sees, OCR noise
included, not hand-typed clean labels.

Usage:
    python scripts/eval_floorplan_classifier.py extract \\
        --images-dir data/images --out eval --eval-fraction 0.4 --seed 42
    # ... hand-label eval/detections.csv's ground_truth_class column ...
    python scripts/eval_floorplan_classifier.py score \\
        --detections eval/detections.csv --out eval/REPORT.md
"""

import argparse
import csv
import json
import os
import random
import re
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

CLASSES = ["kitchens", "bathrooms", "halls", "rooms", "others"]


def _split(image_files: list, eval_fraction: float, seed: int) -> tuple:
    shuffled = sorted(image_files)  # sort first so shuffle is deterministic across OSes
    random.Random(seed).shuffle(shuffled)
    n_eval = round(len(shuffled) * eval_fraction)
    eval_set = sorted(shuffled[:n_eval])
    dev_set = sorted(shuffled[n_eval:])
    return eval_set, dev_set


def cmd_extract(args):
    import numpy as np
    from PIL import Image
    from ultralytics import YOLO
    import easyocr

    from src.floorplan import FLOORPLAN_MODEL_PATH, classify_room_label

    image_files = [
        f for f in sorted(os.listdir(args.images_dir)) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    eval_set, dev_set = _split(image_files, args.eval_fraction, args.seed)

    os.makedirs(args.out, exist_ok=True)
    crops_dir = os.path.join(args.out, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    split_path = os.path.join(args.out, "split.json")
    with open(split_path, "w") as f:
        json.dump(
            {
                "seed": args.seed,
                "eval_fraction": args.eval_fraction,
                "total_images": len(image_files),
                "eval_images": eval_set,
                "dev_images": dev_set,
            },
            f,
            indent=2,
        )
    print(f"Split {len(image_files)} images: {len(eval_set)} eval / {len(dev_set)} dev. Wrote {split_path}")

    print(f"Loading YOLO model from {FLOORPLAN_MODEL_PATH}...")
    model = YOLO(FLOORPLAN_MODEL_PATH)
    print("Loading EasyOCR reader...")
    reader = easyocr.Reader(["en"])

    rows = []
    for image_name in eval_set:
        image_path = os.path.join(args.images_dir, image_name)
        img_pil = Image.open(image_path).convert("RGB")
        results = model.predict(img_pil, imgsz=640, conf=0.25, verbose=False)
        result = results[0]
        class_names = result.names

        box_index = 0
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_names[class_id] != "room_name":
                    continue

                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords
                crop = img_pil.crop((x1, y1, x2, y2))

                crop_file = f"{os.path.splitext(image_name)[0]}_box{box_index}.jpg"
                crop.save(os.path.join(crops_dir, crop_file))

                ocr_result_list = reader.readtext(np.array(crop), detail=0)
                raw_text = " ".join(ocr_result_list)
                cleaned_text = re.sub(r"[^a-z\s]", "", raw_text.lower()).strip()
                predicted_class = classify_room_label(cleaned_text)

                rows.append(
                    {
                        "image": image_name,
                        "box_index": box_index,
                        "crop_file": crop_file,
                        "raw_ocr_text": raw_text,
                        "cleaned_text": cleaned_text,
                        "predicted_class": predicted_class,
                        "ground_truth_class": "",
                    }
                )
                box_index += 1

        print(f"{image_name}: {box_index} room_name box(es)")

    detections_path = os.path.join(args.out, "detections.csv")
    with open(detections_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "box_index",
                "crop_file",
                "raw_ocr_text",
                "cleaned_text",
                "predicted_class",
                "ground_truth_class",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} detected box(es) to {detections_path}")
    print(f"Crops saved under {crops_dir}/ -- fill in ground_truth_class for each row, then run 'score'.")


def cmd_score(args):
    with open(args.detections, newline="") as f:
        rows = list(csv.DictReader(f))

    unlabeled = [r for r in rows if not r["ground_truth_class"].strip()]
    if unlabeled:
        print(f"ERROR: {len(unlabeled)}/{len(rows)} row(s) have no ground_truth_class. Label them first.")
        for r in unlabeled[:10]:
            print(f"  {r['image']} box {r['box_index']} ({r['crop_file']})")
        return 1

    bad_labels = sorted({r["ground_truth_class"] for r in rows} - set(CLASSES))
    if bad_labels:
        print(f"ERROR: unrecognized ground_truth_class value(s): {bad_labels}. Must be one of {CLASSES}.")
        return 1

    # confusion[true_class][predicted_class] = count
    confusion = {t: {p: 0 for p in CLASSES} for t in CLASSES}
    for r in rows:
        confusion[r["ground_truth_class"]][r["predicted_class"]] += 1

    lines = []
    lines.append("# Floorplan room-type classifier evaluation\n")
    lines.append(f"Reproduce with: `{args.repro_command}`\n")
    lines.append(f"N = {len(rows)} detected `room_name` boxes across the held-out eval split.\n")

    lines.append("## Confusion matrix (rows = ground truth, columns = predicted)\n")
    header = "| true \\ pred | " + " | ".join(CLASSES) + " | support |"
    sep = "|---" * (len(CLASSES) + 2) + "|"
    lines.append(header)
    lines.append(sep)
    for t in CLASSES:
        support = sum(confusion[t].values())
        cells = " | ".join(str(confusion[t][p]) for p in CLASSES)
        lines.append(f"| **{t}** | {cells} | {support} |")
    lines.append("")

    lines.append("## Per-class precision / recall\n")
    lines.append("| class | precision | recall | support (true) | predicted count |")
    lines.append("|---|---|---|---|---|")
    for c in CLASSES:
        tp = confusion[c][c]
        support = sum(confusion[c].values())  # all true instances of c
        predicted_count = sum(confusion[t][c] for t in CLASSES)  # all predictions of c
        precision = tp / predicted_count if predicted_count else float("nan")
        recall = tp / support if support else float("nan")
        p_str = f"{precision:.2f}" if predicted_count else "n/a (0 predicted)"
        r_str = f"{recall:.2f}" if support else "n/a (0 true)"
        lines.append(f"| {c} | {p_str} | {r_str} | {support} | {predicted_count} |")
    lines.append("")

    total = len(rows)
    correct = sum(confusion[c][c] for c in CLASSES)
    accuracy = correct / total if total else float("nan")
    lines.append(f"Overall accuracy: {correct}/{total} = {accuracy:.2f}\n")

    report = "\n".join(lines)
    print(report)

    if args.out:
        with open(args.out, "w") as f:
            f.write(report)
        print(f"\nWrote {args.out}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser(
        "extract", help="Run the real pipeline on a held-out split, write a labeling CSV"
    )
    p_extract.add_argument("--images-dir", default="data/images")
    p_extract.add_argument("--out", default="eval")
    p_extract.add_argument("--eval-fraction", type=float, default=0.4)
    p_extract.add_argument("--seed", type=int, default=42)

    p_score = sub.add_parser("score", help="Compute the confusion matrix and per-class precision/recall")
    p_score.add_argument("--detections", default="eval/detections.csv")
    p_score.add_argument("--out", default="eval/REPORT.md")
    p_score.add_argument(
        "--repro-command",
        default="python scripts/eval_floorplan_classifier.py extract --eval-fraction 0.4 --seed 42",
    )

    args = parser.parse_args()
    if args.command == "extract":
        return cmd_extract(args) or 0
    return cmd_score(args)


if __name__ == "__main__":
    sys.exit(main())
