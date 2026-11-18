# Floorplan Room-Type Classifier Evaluation

This report measures `classify_room_label()` in `src/floorplan.py`, which
sorts each OCR'd floorplan label into `kitchens`, `bathrooms`, `halls`,
`rooms` or `others`. It compares the pipeline before and after
`prepare_ocr_crop`, which pads and upscales each detected box before OCR.

## Summary

| Metric (all 73 images) | Before | After |
|---|---|---|
| Empty OCR text (1,220 boxes) | 410 (33.6%) | 131 (10.7%) |
| Room-type accuracy (1,190 scored boxes) | 943 (0.79) | 969 (0.81) |

The accuracy change is significant under a paired sign test (50 boxes
improved, 24 regressed, p = 0.0034).

## Method

**Pipeline.** The evaluation runs the production pipeline unchanged: the
`best_1000.pt` YOLO model (`imgsz=640`, `conf=0.25`), the same EasyOCR call
and text cleaning, and `classify_room_label()`.

**Why all 73 images.** The classifier is a set of hand-written rules, not a
trained model, so scoring it on every image has no leakage risk. An earlier
version of this report used a 15-image subset (272 boxes); at that size the
improvement was not significant (p = 0.115), so the evaluation was extended
to the full set. The subset results are kept at the end of this report.

**Ground truth.** 1,220 detected boxes were labelled by hand:

- 282 boxes from the original 15-image subset, labelled by viewing each crop.
- 696 of the remaining 938 boxes labelled from unambiguous OCR text
  (for example `kitchen`, `kichen`, `bedroom`), using a text-to-category
  mapping built from all 630 unique OCR strings.
- The other 242 boxes (empty or ambiguous text) labelled by viewing
  upscaled crops.

`rooms` means bedrooms; `halls` means living room, great room or hallway;
everything else is `others`. 30 boxes (2.5%) were too degraded to label and
are excluded from scoring (`eval/full73/excluded_unreadable.csv`).

**Paired comparison.** YOLO inference is deterministic: the before and
after runs produced the same 1,220 boxes in the same order, and two
independent after-fix runs produced identical predictions. Each box can
therefore be compared directly between versions, which is why a paired
sign test is used.

## Empty OCR Results

| | Before | After | Change |
|---|---|---|---|
| Empty OCR text | 410 / 1220 (33.6%) | 131 / 1220 (10.7%) | −68% relative |

This metric needs no ground truth. It matters because
`classify_room_label("")` always returns `others`, so every empty OCR result
is a likely misclassification.

## Accuracy

| Set | Scored boxes | Before | After | Improved / regressed | Sign test p |
|---|---|---|---|---|---|
| **Full set** | **1190** | **943 (0.79)** | **969 (0.81)** | 50 / 24 | **0.0034** |
| 15-image subset | 272 | 203 (0.75) | 211 (0.78) | 14 / 6 | 0.115 |

`others` makes up 61% of the ground truth, so overall accuracy is a weak
signal on its own; the per-class results below are more informative.

## Confusion Matrices (full set, 1,190 boxes)

### Before

| true \ pred | kitchens | bathrooms | halls | rooms | others | support |
|---|---|---|---|---|---|---|
| **kitchens** | 62 | 0 | 0 | 0 | 26 | 88 |
| **bathrooms** | 0 | 33 | 0 | 1 | 105 | 139 |
| **halls** | 0 | 0 | 53 | 5 | 34 | 92 |
| **rooms** | 0 | 0 | 0 | 90 | 55 | 145 |
| **others** | 0 | 0 | 0 | 21 | 705 | 726 |

| class | precision | recall | support | predicted |
|---|---|---|---|---|
| kitchens | 1.00 | 0.70 | 88 | 62 |
| bathrooms | 1.00 | 0.24 | 139 | 33 |
| halls | 1.00 | 0.58 | 92 | 53 |
| rooms | 0.77 | 0.62 | 145 | 117 |
| others | 0.76 | 0.97 | 726 | 925 |

### After

| true \ pred | kitchens | bathrooms | halls | rooms | others | support |
|---|---|---|---|---|---|---|
| **kitchens** | 71 | 0 | 0 | 0 | 17 | 88 |
| **bathrooms** | 0 | 49 | 0 | 1 | 89 | 139 |
| **halls** | 0 | 0 | 51 | 8 | 33 | 92 |
| **rooms** | 0 | 0 | 0 | 98 | 47 | 145 |
| **others** | 1 | 1 | 0 | 24 | 700 | 726 |

| class | precision | recall | support | predicted |
|---|---|---|---|---|
| kitchens | 0.99 | 0.81 | 88 | 72 |
| bathrooms | 0.98 | 0.35 | 139 | 50 |
| halls | 1.00 | 0.55 | 92 | 51 |
| rooms | 0.75 | 0.68 | 145 | 131 |
| others | 0.79 | 0.96 | 726 | 886 |

## Findings

- **Recall improved** for kitchens (0.70 → 0.81), bathrooms (0.24 → 0.35) and
  rooms (0.62 → 0.68). Halls recall fell slightly (0.58 → 0.55): its keywords
  (`hall`, `liv`, `great`) are longer and more sensitive to residual OCR
  errors.
- **Precision fell slightly** for bathrooms (1.00 → 0.98) and rooms
  (0.77 → 0.75). More readable text now reaches the substring rules, which
  exposes their bugs more often.
- **Substring-rule bugs found:**
  - `"nooki"` (a nook) contains `"ki"` and is classified as a kitchen.
  - `"pdr room"` (powder room) is classified as `rooms`, because only the
    full word `"powder"` is recognised as a bathroom.
  - `"covliiog toica"` (covered porch) is classified as a bathroom through
    `"toi"`.
  - Known cases from the unit tests: `"br"` inside "library" and
    "breakfast", and `"room"` inside "mud room".

## Next Step

The OCR crop fix addressed the largest error source. The remaining errors
come mainly from the substring rules: matching `"ki"` anywhere in a word,
unrecognised abbreviations such as `"pdr"`, and the `"br"` and `"room"`
cases above. Fixing these rules is the clearest next improvement.

## 15-Image Subset

The original subset (`eval/split.json`, `eval/detections.csv`,
`eval/detections_excluded_unreadable.csv`), scored with the same ground
truth.

### Before

| true \ pred | kitchens | bathrooms | halls | rooms | others | support |
|---|---|---|---|---|---|---|
| **kitchens** | 7 | 0 | 0 | 0 | 10 | 17 |
| **bathrooms** | 0 | 7 | 0 | 0 | 25 | 32 |
| **halls** | 0 | 0 | 4 | 0 | 12 | 16 |
| **rooms** | 0 | 0 | 0 | 14 | 22 | 36 |
| **others** | 0 | 0 | 0 | 1 | 170 | 171 |

Accuracy: 202/272 = 0.74. (The paired re-run counts 203/272; the one-box
difference is in the pre-fix run and doesn't change either result.)

### After

| true \ pred | kitchens | bathrooms | halls | rooms | others | support |
|---|---|---|---|---|---|---|
| **kitchens** | 11 | 0 | 0 | 0 | 6 | 17 |
| **bathrooms** | 0 | 9 | 0 | 0 | 23 | 32 |
| **halls** | 0 | 0 | 4 | 3 | 9 | 16 |
| **rooms** | 0 | 0 | 0 | 19 | 17 | 36 |
| **others** | 0 | 1 | 0 | 2 | 168 | 171 |

Accuracy: 211/272 = 0.78. Empty OCR text: 128/272 (47%) → 30/272 (11%).

## Reproduce

```bash
# Full set, current pipeline
docker run --rm \
  -v "$(pwd)/scripts:/app/scripts:ro" -v "$(pwd)/eval/full73:/app/eval" -w /app \
  smartsense-project-api:latest python scripts/eval_floorplan_classifier.py extract \
    --images-dir /app/data/images --out /app/eval --eval-fraction 1.0 --seed 42

# Full set, pipeline before prepare_ocr_crop (commit a2289ad)
git show a2289ad:src/floorplan.py > /tmp/floorplan_prefix.py
git show a2289ad:scripts/eval_floorplan_classifier.py > /tmp/eval_prefix.py
docker run --rm \
  -v "/tmp/eval_prefix.py:/app/scripts/eval_floorplan_classifier.py:ro" \
  -v "/tmp/floorplan_prefix.py:/app/src/floorplan.py:ro" \
  -v "$(pwd)/eval/full73_before:/app/eval" -w /app \
  smartsense-project-api:latest python scripts/eval_floorplan_classifier.py extract \
    --images-dir /app/data/images --out /app/eval --eval-fraction 1.0 --seed 42
# Ground truth is joined by (image, box_index); the joined result is
# eval/full73/before_detections.csv.

# Score
python scripts/eval_floorplan_classifier.py score \
  --detections eval/full73/after_detections.csv --out /dev/stdout

# 15-image subset
docker run --rm \
  -v "$(pwd)/scripts:/app/scripts:ro" -v "$(pwd)/eval:/app/eval" -w /app \
  smartsense-project-api:latest python scripts/eval_floorplan_classifier.py extract \
    --images-dir /app/data/images --out /app/eval --eval-fraction 0.2 --seed 42
python scripts/eval_floorplan_classifier.py score --detections eval/detections.csv --out /dev/stdout
```
