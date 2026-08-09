# Floorplan room-type classifier evaluation (Tier 4)

Measures `classify_room_label()` in `src/floorplan.py` — the substring
cascade that buckets each OCR'd floorplan label into `kitchens` /
`bathrooms` / `halls` / `rooms` / `others` — against real detections on a
held-out split of `data/images/`.

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
images (138 boxes had empty OCR text; 128 of those were labelable by eye,
10 were not).

## Confusion matrix (rows = ground truth, columns = predicted)

| true \ pred | kitchens | bathrooms | halls | rooms | others | support |
|---|---|---|---|---|---|---|
| **kitchens** | 7 | 0 | 0 | 0 | 10 | 17 |
| **bathrooms** | 0 | 7 | 0 | 0 | 25 | 32 |
| **halls** | 0 | 0 | 4 | 0 | 11 | 15 |
| **rooms** | 0 | 0 | 0 | 14 | 22 | 36 |
| **others** | 0 | 0 | 0 | 1 | 171 | 172 |

## Per-class precision / recall

| class | precision | recall | support (true) | predicted count |
|---|---|---|---|---|
| kitchens | 1.00 | 0.41 | 17 | 7 |
| bathrooms | 1.00 | 0.22 | 32 | 7 |
| halls | 1.00 | 0.27 | 15 | 4 |
| rooms | 0.93 | 0.39 | 36 | 15 |
| others | 0.72 | 0.99 | 172 | 239 |

Overall accuracy: 203/272 = 0.75 (dominated by `others`, which is 63% of
the sample by ground truth — not a meaningful headline number on its own,
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

## Investigation 2: where the real errors come from (this is the important part)

68 of the 100 true kitchen/bathroom/hall/room instances were
misclassified (the rest landed correctly on the diagonal). Every single
one of those 68 traces back to the OCR step, not to
`classify_room_label`'s branch logic:

- **41/68** had **completely empty** OCR text. `classify_room_label("")`
  always falls through to `others` by construction — there's no
  substring for it to possibly match, so these were never in the
  classifier's control at all.
- **27/68** had **non-empty but corrupted** OCR text where the corruption
  specifically destroyed the trigger substring — e.g. `"kitchen"` read
  back as `"ktchey"` (the `i` is gone, so `"ki"` no longer appears), or
  `"bath"` read back as `"batn"` / `"tond"` / `"da"`. A human reading the
  crop can tell these say KITCHEN and BATH; the substring rule, working
  correctly on the (wrong) text it was actually given, cannot.

**Zero of the 68** were a case of "OCR read the label correctly and the
substring cascade still picked the wrong bucket" — the specific failure
mode Tier 4 flagged and that a rule fix or a replacement classifier would
actually target.

Cross-checked against crop geometry: boxes with empty OCR text average
28×14px; boxes OCR successfully read average 57×17px. Visual review
confirms this isn't "the text was genuinely illegible" — `"DINING"`,
`"BATH"`, `"CARPORT"`, `"WIC"` are all clearly readable by eye at 5×
upscale on crops where EasyOCR returned nothing. **The bottleneck is OCR
recall on small crops, not the classification rules.**

The one exception, and it's a real rule bug independent of OCR quality:
`"mueroom"` (a garbled OCR read of "MUD ROOM") contains the literal
substring `"room"`, so it's classified `rooms` — but a *perfectly* OCR'd
`"mud room"` would trigger the exact same false match, since the rule
can't distinguish "mud room" from "bedroom." This is the one true
substring-logic error in the sample (true `others`, predicted `rooms`),
as opposed to the 68 OCR-driven ones above.

## Recommendation (not implemented — for sign-off)

**Neither fixing the substring rules nor replacing them with a small
classifier is the highest-leverage next step**, based on this evidence.
Both approaches still take OCR text as input, and in this sample 100% of
the classifier's errors on real categories happened upstream of
`classify_room_label` — in OCR recall on crops that are frequently too
small (avg. 28×14px) for EasyOCR to return anything, even when a human
reads them fine. Fixing or replacing the classifier would correctly
handle the *one* observed rule-logic bug (`"room"` inside "mud room") and
would harden against the *documented-but-not-observed-here* bugs
(parking/library/breakfast) — real, but secondary, wins.

If you want to spend Tier 4 effort where this evidence says it pays off
most: look at the crop/OCR step first — larger crop padding, upscaling
crops before OCR, or an OCR confidence/preprocessing change — and re-run
this same eval afterward to see how much of the 68 that unblocks. Once
OCR recall is no longer eating the majority of the signal, a rule fix
(reorder / tighten the substring checks) is cheap and would close the
`"mud room"`-style gap plus the untriggered parking/library/breakfast
bugs. A learned classifier's case gets stronger mainly if you want
robustness against *arbitrary* future vocabulary rather than the specific
known failure modes — worth deciding once the OCR bottleneck is
addressed, not before.

Your call on which of these to pursue, if any.
