# Floorplan room-type classifier evaluation (Tier 4)

Measures `classify_room_label()` in `src/floorplan.py` — the substring
cascade that buckets each OCR'd floorplan label into `kitchens` /
`bathrooms` / `halls` / `rooms` / `others` — against real detections on a
held-out split of `data/images/`. Baseline measurement below is followed
by an implemented fix (`prepare_ocr_crop`) and a re-measurement, per the
baseline's own recommendation.

## Reproduce

```bash
# 1. Run the real YOLO + EasyOCR pipeline over the held-out split, write a
#    labeling worksheet (needs the app image: torch/ultralytics/easyocr)
docker run --rm \
  -v "$(pwd)/scripts:/app/scripts:ro" -v "$(pwd)/eval:/app/eval" -w /app \
  smartsense-project-app:latest python scripts/eval_floorplan_classifier.py extract \
    --images-dir /app/data/images --out /app/eval --eval-fraction 0.2 --seed 42

# 2. Hand-label eval/detections.csv's ground_truth_class column (already
#    done -- see Methodology below -- this step is committed, not rerun)

# 3. Score
python scripts/eval_floorplan_classifier.py score \
  --detections eval/detections.csv --out eval/REPORT.md
```

## Methodology

**What was held out, from what.** `data/images/` has 73 labeled floorplan
images. A fixed-seed (42) shuffle splits them into 15 **eval** images
(20%) and 58 **dev** images (80%) — see `eval/split.json` for the exact
list. Every number below comes only from the 15 eval images; the 58 dev
images were never looked at while producing these numbers and are
reserved for future rule iteration if Tier 4 proceeds past measurement.

`classify_room_label()` itself is fixed hand-written rules, not something
trained on this data, so there's no train/test leakage risk in the usual
ML sense. The split still matters for two reasons: it stops this report
from being tuned (consciously or not) against the same examples it's
graded on, and it's the only honest way to state a sample size.

**Limitation, stated plainly:** the YOLO detector *was* trained on a
subset of this image pool (`notebooks/train.ipynb`, 80/20 split,
`random_state=42`), but its original COCO annotation file lived outside
this repo (Google Drive) and isn't recoverable from what's checked in —
so it's not possible to guarantee these 15 eval images were unseen by the
detector during *its* training. If any were in YOLO's training fold,
detection recall (how much text reaches the classifier at all) may be
mildly optimistic here relative to genuinely novel floorplans. This
doesn't affect the *classifier's* precision/recall given a detected box —
only how representative the box population itself is.

**What's actually measured.** The real pipeline, not a synthetic
approximation: the same `best_1000.pt` YOLO model at `imgsz=640,
conf=0.25`, the same EasyOCR call, the same regex cleaning
(`src/floorplan.py`'s `parse_floorplan`), feeding the real (and
sometimes wrong) `detected_text` into the actual, unmodified
`classify_room_label()`. OCR noise is part of what's being measured, not
abstracted away — that's deliberate, since it's what the deployed system
actually sees.

**Ground truth.** Every detected `room_name` box in the eval split was
hand-labeled by viewing its cropped image (`eval/crops/`, batched into
upscaled contact sheets for review) against the same 5-bucket taxonomy
the app uses. Two judgment calls, stated explicitly so they can be
second-guessed:

- `rooms` = bedrooms specifically, matching `classify_room_label`'s own
  docstring ("e.g., bedroom") — "bonus room" and "guest suite" count,
  "mud room" and "dining room" don't.
- `halls` = living room / great room / hallway, matching the rule's own
  target keywords (`hall`/`liv`/`great`). Foyers and entries are *not*
  counted as halls even though they're entry-adjacent spaces — the rule
  doesn't target them, so grading them as halls would be scoring the
  classifier against a taxonomy it was never written to hit.

Everything else (garage, patio, porch, closet, storage, utility,
dining, study/office, deck, mudroom, laundry) is `others`.

10 of 282 detected boxes (3.5%) were too degraded to confidently label
even by eye at 5x upscale — excluded from scoring rather than guessed at.
See `eval/detections_excluded_unreadable.csv`.

**Sample size: N = 272** labeled detections across the 15 held-out
images.

**Correction found during the re-verification pass below:** one box
(`19_17_...`, box 2, "GREAT ROOM") was mislabeled `others` in the
original pass — a transcription slip while reading a 30-crop contact
sheet, not a genuine ambiguity. Caught while re-checking a handful of
crops after implementing the fix below, and corrected in both the
baseline and post-fix datasets before any of the numbers in this report
were computed. Flagging it rather than quietly fixing it, since "trust
but verify your own labels too" is the same discipline this whole
exercise is built on.

## Baseline confusion matrix (rows = ground truth, columns = predicted)

| true \ pred | kitchens | bathrooms | halls | rooms | others | support |
|---|---|---|---|---|---|---|
| **kitchens** | 7 | 0 | 0 | 0 | 10 | 17 |
| **bathrooms** | 0 | 7 | 0 | 0 | 25 | 32 |
| **halls** | 0 | 0 | 4 | 0 | 12 | 16 |
| **rooms** | 0 | 0 | 0 | 14 | 22 | 36 |
| **others** | 0 | 0 | 0 | 1 | 170 | 171 |

## Baseline per-class precision / recall

| class | precision | recall | support (true) | predicted count |
|---|---|---|---|---|
| kitchens | 1.00 | 0.41 | 17 | 7 |
| bathrooms | 1.00 | 0.22 | 32 | 7 |
| halls | 1.00 | 0.25 | 16 | 4 |
| rooms | 0.93 | 0.39 | 36 | 15 |
| others | 0.71 | 0.99 | 171 | 239 |

Baseline accuracy: 202/272 = 0.74 (dominated by `others`, 63% of the
sample by ground truth — not a meaningful headline number on its own,
which is exactly why per-class numbers were asked for).

## Investigation 1: the 1.00 precisions are not evidence the rules are fine

Perfect precision on 3 of 4 real classes looks too good given Tier 4's
own bug list (`"ki"` matches *parking*, `"br"` matches *library* and
*breakfast*). It is too good, for a specific, checkable reason: **none of
those trigger words appear anywhere in this 15-image sample.** These
floorplans label vehicle storage "GARAGE" or "CARPORT", never "PARKING",
and none of them have a library, study-as-library, or breakfast nook
labeled that way. A different 15-image sample, or the 58 dev images, would
very plausibly surface both bugs and pull kitchen/room precision below
1.00. Treat the 1.00s as "not contradicted by this sample," not as "the
documented bugs don't matter."

## Investigation 2: where the baseline errors come from

70 of the 101 true kitchen/bathroom/hall/room instances were
misclassified at baseline. All but one trace back to the OCR step, not to
`classify_room_label`'s branch logic:

- **41 had completely empty OCR text.** `classify_room_label("")` always
  falls through to `others` by construction — there's no substring for it
  to possibly match, so these were never in the classifier's control at
  all.
- **28 had non-empty but corrupted OCR text** where the corruption
  specifically destroyed the trigger substring — e.g. `"kitchen"` read
  back as `"ktchey"` (the `i` is gone, so `"ki"` no longer appears), or
  `"bath"` read back as `"batn"` / `"tond"` / `"da"`. A human reading the
  crop can tell these say KITCHEN and BATH; the substring rule, working
  correctly on the (wrong) text it was actually given, cannot.
- **1** (`"mueroom"`, a garbled OCR read of "MUD ROOM") is a genuine
  rule-logic bug independent of OCR quality: it contains the literal
  substring `"room"`, so it's classified `rooms` — but a *perfectly*
  OCR'd `"mud room"` would trigger the exact same false match, since the
  rule can't distinguish "mud room" from "bedroom."

Cross-checked against crop geometry: boxes with empty OCR text averaged
28×14px; boxes OCR successfully read averaged 57×17px. Visual review
confirmed this wasn't "the text was illegible" — `"DINING"`, `"BATH"`,
`"CARPORT"`, `"WIC"` were all clearly readable by eye at 5× upscale on
crops where EasyOCR returned nothing. **Baseline conclusion: the
bottleneck was OCR recall on small crops, not the classification rules.**

## Implemented: `prepare_ocr_crop` (padding + upscaling before OCR)

Per the baseline's own recommendation ("look at the crop/OCR step first
... re-run this same eval afterward"), added `prepare_ocr_crop()` in
`src/floorplan.py`, wired into `parse_floorplan()` in place of the bare
`img_pil.crop(...)`:

- **Pad** each detected box by 15% a side (min 2px) before cropping — a
  tight YOLO box can clip a character's edge.
- **Upscale** (LANCZOS) so the crop is at least 48px tall, capped at 6x
  to avoid blowing up already-adequate crops into blur.

`scripts/eval_floorplan_classifier.py` was updated to call the same
function, so the eval and the production pipeline share one
implementation rather than two that could drift apart — same principle
already applied to `classify_room_label`.

**Re-measured on the identical held-out split** (same 15 images, same
seed; box detection is deterministic so the same 282 boxes came back in
the same order — ground truth was carried over by `(image, box_index)`,
not relabeled from scratch).

### Post-fix confusion matrix

| true \ pred | kitchens | bathrooms | halls | rooms | others | support |
|---|---|---|---|---|---|---|
| **kitchens** | 11 | 0 | 0 | 0 | 6 | 17 |
| **bathrooms** | 0 | 9 | 0 | 0 | 23 | 32 |
| **halls** | 0 | 0 | 4 | 3 | 9 | 16 |
| **rooms** | 0 | 0 | 0 | 19 | 17 | 36 |
| **others** | 0 | 1 | 0 | 2 | 168 | 171 |

### Post-fix per-class precision / recall, next to baseline

| class | precision (before → after) | recall (before → after) | support |
|---|---|---|---|
| kitchens | 1.00 → 1.00 | 0.41 → **0.65** | 17 |
| bathrooms | 1.00 → 0.90 | 0.22 → 0.28 | 32 |
| halls | 1.00 → 1.00 | 0.25 → 0.25 | 16 |
| rooms | 0.93 → 0.79 | 0.39 → **0.53** | 36 |
| others | 0.71 → 0.75 | 0.99 → 0.98 | 171 |

Accuracy: 202/272 (0.74) → **211/272 (0.78)**.

Empty-OCR-text rate: **128/272 (47%) → 30/272 (11%)** — a 77% relative
reduction, and the clearest confirmation the fix hit its actual target.

## Investigation 3: the fix worked, but unevenly — and it has a cost

**Where it clearly helped:** kitchens and rooms both gained meaningfully
on recall (+0.24, +0.14) with kitchens holding perfect precision. Both
rely on short, common trigger substrings (`"ki"`, `"bed"`/`"br"`) that
survive minor residual OCR noise as long as *some* legible text comes
back at all — which is now far more often true.

**Where it didn't: halls stayed at 0.25 recall, unchanged.** Checking why
against the actual post-fix OCR output for the 16 true-hall boxes: OCR
recall genuinely improved here too (almost none are empty anymore), but
`"hall"` / `"liv"` / `"great"` are longer, pickier trigger substrings than
`"ki"` or `"bath"`, and the residual character-level noise that padding
and upscaling didn't eliminate keeps landing on exactly those letters:
`"living room"` → `"lming"` / `"lming room"` / `"lng"` (no `"liv"`
survives), `"great room"` → `"geat room"` / `"grealrcom"` / `"gruid"` (no
`"great"` survives), `"hallway"` → `"mallway"` (the leading `h` is gone,
so no `"hall"`). The fix increased *how often* legible-ish text comes
back; it didn't fully solve *character-level* accuracy, and halls'
trigger words are the least tolerant of the two.

**The cost: precision dropped for bathrooms (1.00→0.90) and rooms
(0.93→0.79).** More text reaching `classify_room_label` instead of
failing silently into `others` also means more chances for the
substring-logic bugs to fire. Every new false positive checked by hand:

- `"mubroom"` / `"mud room"` (2 instances) — the same `"mud room"`-contains-
  `"room"` bug identified at baseline, now surfacing twice instead of once
  because OCR actually reads "MUD ROOM" now instead of returning nothing.
- `"lming room"` and `"geat room"` — cases where OCR recovered *enough*
  text to trigger the generic `"room"` fallback but not enough to trigger
  the more specific `"liv"`/`"great"` hall check that should have won.
- `"covliiog toica"` (true: "COVERED PORCH") — a still-garbled OCR read
  that happens to contain `"toi"`, misfiring into bathrooms.

**Conclusion: this is exactly the handoff the baseline predicted.**
Fixing OCR recall was the right first move (77% fewer total failures,
real recall gains on 2 of 4 classes) — but it also unblocked the
substring-logic bugs to matter more, since they can now only misfire on
text that actually comes back. The next highest-leverage step is now the
rules themselves: fixing the specific bugs on record (`"ki"`/parking,
`"br"`/library-breakfast, `"room"`-inside-"mud room") and hardening the
hall keywords against the character-level noise shown above would very
plausibly close a meaningful chunk of the remaining gap, particularly for
halls and rooms precision. Not implemented here — this was the OCR-layer
recommendation specifically; the substring rules are still "don't touch
until you decide," per the original brief.

## Reproduce the post-fix numbers

```bash
docker compose build app   # picks up prepare_ocr_crop
docker run --rm \
  -v "$(pwd)/scripts:/app/scripts:ro" -v "$(pwd)/eval:/app/eval" -w /app \
  smartsense-project-app:latest python scripts/eval_floorplan_classifier.py extract \
    --images-dir /app/data/images --out /app/eval --eval-fraction 0.2 --seed 42
python scripts/eval_floorplan_classifier.py score --detections eval/detections.csv --out eval/REPORT.md
```
