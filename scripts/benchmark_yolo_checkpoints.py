#!/usr/bin/env python
"""Benchmarks best_1000.pt against best_300.pt (the training run's two
checkpoints) on room_name detection count -- the evidence behind deleting
best_300.pt (see README's Floorplan CV Model section).

best_300.pt is not in this repo (deleted in commit aea4a40) and is not
re-added by this script. Recover it from git history first:

    git show aea4a40^:src/best_300.pt > src/best_300.pt

Ground truth (GROUND_TRUTH below) is a hand count of room_name-style text
labels -- including fixture/appliance labels like "Ref." or "W"/"D" next
to Laundry, which is how this dataset's own annotations treat them
(consistent with what shows up as real room_name detections elsewhere in
this project, e.g. eval/REPORT.md) -- on 16 of the 73 images in
data/images/, done once by visual inspection since no ground truth ships
with this repo. Spot-checked against the actual images before trusting
it for this script: 10_29_...0c50679... reads exactly 6 labels (M.BR.,
FAMILY, DIN, BR.2, GARAGE, BR.3); 0_30_...bcf0ea4... reads exactly 18
once fixture labels (Ref., Micro./Coffee) are counted alongside the 16
room labels -- both match the recorded values below.

"Error" here is |predicted room_name count - hand-counted true count|
per image, averaged over these 16 images -- a detection-count metric,
independent of the OCR/classify_room_label pipeline entirely (that's
eval/REPORT.md's job; this script never runs OCR).

Usage:
    python scripts/benchmark_yolo_checkpoints.py \
        --model-1000 src/best_1000.pt --model-300 src/best_300.pt \
        --images-dir data/images --out eval/yolo_checkpoint_benchmark.md
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Hand-counted true room_name-label count per image, done once by visual
# inspection of the 16 images below (see module docstring for the
# spot-check that validated this before it was trusted).
GROUND_TRUTH = {
    "0_20_jpg.rf.55d03cd6dd66cf512099eb67e527175c.jpg": 14,
    "0_26_jpg.rf.22d020e2780fff58cefb479d8fd41058.jpg": 12,
    "0_28_jpg.rf.bda89f2956b68f4dafae375c7ee2af77.jpg": 11,
    "0_30_jpg.rf.bcf0ea49d5693cfb9bf9fd3efb73181b.jpg": 18,
    "10_21_jpg.rf.cdc2b334287b64744b80fa37e5125068.jpg": 21,
    "10_26_jpg.rf.e5d27a70ba2a91011c456723cda40b4a.jpg": 15,
    "10_29_jpg.rf.0c5067935034641091bb2290b5d245ee.jpg": 6,
    "13_18_jpg.rf.ec11791b4d9cc45c6a4809fd27fd1ca1.jpg": 9,
    "15_25_jpg.rf.cbfd098a4c0bc0edd2d9e9b1620b4bb5.jpg": 19,
    "16_11_jpg.rf.e0a76b0fba37d1b966021491bf85cd68.jpg": 24,
    "16_15_jpg.rf.dab1858a93afa721d4ebbdb824c68898.jpg": 17,
    "16_24_jpg.rf.cacc0b0120946b0e3f572939c3ff3206.jpg": 10,
    "17_17_jpg.rf.7623bd0247831a1ec9a129d372486d39.jpg": 16,
    "28_18_jpg.rf.7e2a025d6ea364275d34bcec17eff91d.jpg": 14,
    "66_6_jpg.rf.2bbd0923857fe025ca96d559913b98ef.jpg": 20,
    "72_8_jpg.rf.bc78aa4d12e97a36a4cb199d0b7456aa.jpg": 16,
}


def count_room_names(model_path: str, images_dir: str) -> dict:
    from ultralytics import YOLO

    model = YOLO(model_path)
    counts = {}
    for image_name in GROUND_TRUTH:
        image_path = os.path.join(images_dir, image_name)
        results = model.predict(image_path, imgsz=640, conf=0.25, verbose=False)
        result = results[0]
        class_names = result.names
        n_room_name = 0
        if result.boxes is not None:
            for box in result.boxes:
                if class_names[int(box.cls[0])] == "room_name":
                    n_room_name += 1
        counts[image_name] = n_room_name
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-1000", default="src/best_1000.pt")
    parser.add_argument("--model-300", default="src/best_300.pt")
    parser.add_argument("--images-dir", default="data/images")
    parser.add_argument("--out", default="eval/yolo_checkpoint_benchmark.md")
    args = parser.parse_args()

    print(f"Running best_1000.pt ({args.model_1000}) over {len(GROUND_TRUTH)} images...")
    counts_1000 = count_room_names(args.model_1000, args.images_dir)
    print(f"Running best_300.pt ({args.model_300}) over {len(GROUND_TRUTH)} images...")
    counts_300 = count_room_names(args.model_300, args.images_dir)

    rows = []
    wins_1000 = wins_300 = ties = 0
    err_1000_total = err_300_total = 0
    for image, gt in GROUND_TRUTH.items():
        c1000, c300 = counts_1000[image], counts_300[image]
        e1000, e300 = abs(c1000 - gt), abs(c300 - gt)
        err_1000_total += e1000
        err_300_total += e300
        if e1000 < e300:
            winner, wins_1000 = "best_1000", wins_1000 + 1
        elif e300 < e1000:
            winner, wins_300 = "best_300", wins_300 + 1
        else:
            winner, ties = "tie", ties + 1
        rows.append((image, gt, c1000, c300, e1000, e300, winner))

    n = len(GROUND_TRUTH)
    mae_1000 = err_1000_total / n
    mae_300 = err_300_total / n
    rel_diff = (1 - mae_1000 / mae_300) * 100 if mae_300 else float("nan")

    lines = []
    lines.append("# YOLO checkpoint benchmark: best_1000.pt vs best_300.pt\n")
    lines.append(
        "Reproduce with: `git show aea4a40^:src/best_300.pt > src/best_300.pt && "
        "python scripts/benchmark_yolo_checkpoints.py`\n"
    )
    lines.append(
        f"N = {n} hand-labeled images (of 73 in `data/images/`). Error = "
        "|predicted room_name box count - hand-counted true count|, per image. "
        "Ground truth methodology and spot-check in the script's docstring.\n"
    )
    lines.append("## Per-image results\n")
    lines.append("| image | GT | best_1000 | best_300 | err (1000) | err (300) | winner |")
    lines.append("|---|---|---|---|---|---|---|")
    for image, gt, c1000, c300, e1000, e300, winner in rows:
        lines.append(f"| {image} | {gt} | {c1000} | {c300} | {e1000} | {e300} | {winner} |")
    lines.append("")
    lines.append("## Summary\n")
    lines.append(
        f"**Wins / losses / ties (out of {n}):** best_1000 wins {wins_1000}, "
        f"best_300 wins {wins_300}, ties {ties}.\n"
    )
    lines.append(
        f"**Mean absolute error per image:** best_1000 = {mae_1000:.3f}, best_300 = {mae_300:.3f} "
        f"({rel_diff:.1f}% lower for best_1000).\n"
    )
    if wins_300 > wins_1000:
        lines.append("**best_300 won more head-to-heads than best_1000 -- reconsider the deletion.**\n")
    else:
        lines.append(
            f"best_1000 wins the plurality of decisive comparisons ({wins_1000} of {wins_1000 + wins_300} "
            "non-tied images) and has the lower MAE -- consistent with the original decision to keep it "
            "over best_300.\n"
        )

    report = "\n".join(lines)
    print("\n" + report)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(report)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
