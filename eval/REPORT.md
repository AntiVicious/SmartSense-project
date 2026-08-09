# Floorplan room-type classifier evaluation (Tier 4)

Measures `classify_room_label()` in `src/floorplan.py` — the substring
cascade that buckets each OCR'd floorplan label into `kitchens` /
`bathrooms` / `halls` / `rooms` / `others` — against real detections.
Baseline measurement, an implemented fix (`prepare_ocr_crop`), and a
re-measurement.

> **Correction (this revision).** Two figures in an earlier version of
> this report didn't hold up to review and are fixed here:
> 1. Accuracy was originally reported on a 15-image held-out subset
>    (N=272 boxes). At that N, each image is worth ~6.7 accuracy points —
>    the reported 0.74→0.78 delta is under one image's worth of signal
>    and isn't distinguishable from noise. Re-run over **all 73 images**
>    (N=1190 scored boxes) below, with a proper paired significance test,
>    not just a before/after point estimate.
> 2. The empty-OCR-rate before/after — measured over crops, not images,
>    so it has ~4.5x more units behind it even at the old N — is promoted
>    to the primary result. It doesn't need ground-truth labeling at all
>    (it's just "did OCR return text or not"), so it's the cheapest number
>    here to trust.
>
> The 15-image subset is kept below, explicitly labeled as the original
> holdout, for transparency about what changed.

## Reproduce

```bash
# Full 73-image set, current (after-fix) pipeline
docker run --rm \
  -v "$(pwd)/scripts:/app/scripts:ro" -v "$(pwd)/eval/full73:/app/eval" -w /app \
  smartsense-project-api:latest python scripts/eval_floorplan_classifier.py extract \
    --images-dir /app/data/images --out /app/eval --eval-fraction 1.0 --seed 42
# -> eval/full73/after_detections.csv (ground truth already filled in, see Methodology)

# Full 73-image set, pre-fix pipeline (for the before/after comparison) --
# retrieves the code as it was immediately before the OCR-crop-padding fix
# (commit ddc16fc) and runs the same extraction with it:
git show a2289ad:src/floorplan.py > /tmp/floorplan_prefix.py
git show a2289ad:scripts/eval_floorplan_classifier.py > /tmp/eval_prefix.py
docker run --rm \
  -v "/tmp/eval_prefix.py:/app/scripts/eval_floorplan_classifier.py:ro" \
  -v "/tmp/floorplan_prefix.py:/app/src/floorplan.py:ro" \
  -v "$(pwd)/eval/full73_before:/app/eval" -w /app \
  smartsense-project-api:latest python scripts/eval_floorplan_classifier.py extract \
    --images-dir /app/data/images --out /app/eval --eval-fraction 1.0 --seed 42
# -> join ground truth from after_detections.csv by (image, box_index); see
#    eval/full73/before_detections.csv for the already-joined result.

# Score either one
python scripts/eval_floorplan_classifier.py score \
  --detections eval/full73/after_detections.csv --out /dev/stdout
```

## Methodology

**Why the full 73 images, not a held-out split, for the headline number.**
`classify_room_label()` is fixed hand-written rules, not something trained
on this data — there's no leakage risk in scoring it on every image, which
is exactly the reasoning that motivated re-running this at N=73 instead of
N=15. The 15-image subset (`eval/split.json`) is kept as a separate,
labeled result below for continuity with the original measurement, not
because it's methodologically necessary anymore.

**What's actually measured.** The real pipeline, not a synthetic
approximation: the same `best_1000.pt` YOLO model at `imgsz=640,
conf=0.25`, the same EasyOCR call, the same regex cleaning
(`src/floorplan.py`'s `parse_floorplan`), feeding the real (and sometimes
wrong) `detected_text` into the actual, unmodified `classify_room_label()`.

**Ground truth, N=1220 detected boxes across 73 images.** Built in two
passes:
- The original 15-image subset's 282 hand-labeled boxes (already
  reviewed, one correction already applied — see the original report's
  note, preserved in git history) carried forward by `(image,
  box_index)` join — not relabeled.
- The remaining 58 images' 938 boxes labeled fresh: 696 resolved directly
  from OCR text via a documented text→category mapping
  (`cleaned_text` values like `"kitchen"`, `"kichen"`, `"bedroom"` are
  unambiguous even garbled) built by reading all 630 unique OCR strings
  in this run; the remaining 242 (every empty-text box plus every
  genuinely ambiguous or garbled non-empty one — nothing was assumed)
  reviewed visually via upscaled contact sheets, same method as the
  original 15-image pass.

Same taxonomy as before: `rooms` = bedrooms specifically; `halls` =
living room / great room / hallway (not foyer, not family room — matching
the substring rule's own target keywords, not a broader "similar space"
reading); everything else is `others`. 30 of 1220 boxes (2.5%) were too
degraded to confidently label even by eye — excluded from scoring, listed
in `eval/full73/excluded_unreadable.csv`.

Box detection is deterministic (same model, same params, no randomness in
inference) — verified directly, not assumed: the after-fix run was
executed twice independently this session and produced byte-identical
predictions on all 272 shared boxes; the before-fix and after-fix runs
produced the same 1220 total boxes in the same order, which is what makes
carrying one ground-truth labeling pass across both valid.

## Primary result: empty-OCR-rate, N = 1220 boxes across all 73 images

| | before `prepare_ocr_crop` | after | change |
|---|---|---|---|
| Empty OCR text | 410 / 1220 (33.6%) | 131 / 1220 (10.7%) | 68% relative reduction |

This needs no ground-truth labeling — it's a direct count of whether
EasyOCR returned any text at all — so it's the number to trust most in
this report. It's also the mechanism behind every other number below:
`classify_room_label("")` always resolves to `others`, so this is the
single biggest lever on the classifier's apparent accuracy, independent
of the substring rules' own correctness.

## Secondary, directional: accuracy, with sample size and a real significance test

Point accuracy alone is a weak signal at any N here (`others` is 61% of
the sample by ground truth) — reported with a **paired** test (same 1190
boxes scored under both pipeline versions; a box flips from wrong→right,
right→wrong, or stays put) rather than treating before/after as
independent samples, since that's what they actually are.

| Split | N (scored boxes) | Accuracy before | Accuracy after | Improved / regressed (of N) | Paired sign-test p-value |
|---|---|---|---|---|---|
| **Full set (headline)** | **1190** | **943/1190 = 0.79** | **969/1190 = 0.81** | 50 improved, 24 regressed (net +26) | **p = 0.0034** |
| 15-image subset (original holdout) | 272 | 203/272 = 0.75 | 211/272 = 0.78 | 14 improved, 6 regressed (net +8) | p = 0.115 |

Reading this straight: at the original N=272, the improvement is **not**
distinguishable from noise (p=0.115) — confirming the original concern
exactly. At the full N=1190, the same kind of test on the same kind of
paired data **is** significant (p=0.0034). The full-set delta (+2
points) is smaller than the 15-image subset's (+3 points) — the original
subset's larger-looking delta was itself partly noise, in the direction
you'd worry about (overstating the effect), not just imprecise.

(15-image subset's before-accuracy here, 203/272, differs from this
subset by one box from the number in the original report, 202/272 —
tracked down to the pre-fix run specifically, not the after-fix run
(which reproduced byte-identical on both independent runs); not chased
further since it doesn't change either number's conclusion or the
paired-test result, which uses this run's own numbers on both sides
consistently.)

## Confusion matrices and per-class precision/recall, full set (N=1190)

### Before (pre-`prepare_ocr_crop`)

| true \ pred | kitchens | bathrooms | halls | rooms | others | support |
|---|---|---|---|---|---|---|
| **kitchens** | 62 | 0 | 0 | 0 | 26 | 88 |
| **bathrooms** | 0 | 33 | 0 | 1 | 105 | 139 |
| **halls** | 0 | 0 | 53 | 5 | 34 | 92 |
| **rooms** | 0 | 0 | 0 | 90 | 55 | 145 |
| **others** | 0 | 0 | 0 | 21 | 705 | 726 |

| class | precision | recall | support (true) | predicted count |
|---|---|---|---|---|
| kitchens | 1.00 | 0.70 | 88 | 62 |
| bathrooms | 1.00 | 0.24 | 139 | 33 |
| halls | 1.00 | 0.58 | 92 | 53 |
| rooms | 0.77 | 0.62 | 145 | 117 |
| others | 0.76 | 0.97 | 726 | 925 |

### After (with `prepare_ocr_crop`)

| true \ pred | kitchens | bathrooms | halls | rooms | others | support |
|---|---|---|---|---|---|---|
| **kitchens** | 71 | 0 | 0 | 0 | 17 | 88 |
| **bathrooms** | 0 | 49 | 0 | 1 | 89 | 139 |
| **halls** | 0 | 0 | 51 | 8 | 33 | 92 |
| **rooms** | 0 | 0 | 0 | 98 | 47 | 145 |
| **others** | 1 | 1 | 0 | 24 | 700 | 726 |

| class | precision | recall | support (true) | predicted count |
|---|---|---|---|---|
| kitchens | 0.99 | 0.81 | 88 | 72 |
| bathrooms | 0.98 | 0.35 | 139 | 50 |
| halls | 1.00 | 0.55 | 92 | 51 |
| rooms | 0.75 | 0.68 | 145 | 131 |
| others | 0.79 | 0.96 | 726 | 886 |

## Investigation: what the larger sample changes about the original findings

**The "1.00 precision" caveat from the 15-image report was correct to
flag, and now confirmed directly.** At N=1190, kitchens precision drops
to 0.99 and bathrooms to 0.98 — both from new false positives that
simply weren't present in the smaller sample:

- `"nooki"` (true: a nook, `others`) contains the literal substring
  `"ki"` → misclassified `kitchens`. This is the exact `"ki"`/parking-
  shaped bug the original report predicted would surface with more data
  — same failure mode, different trigger word.
- `"pdr room"` (true: a powder room, `bathrooms`) → misclassified
  `rooms`. A previously undocumented gap: `classify_room_label` only
  recognizes the literal substring `"powder"` for bathrooms, not the
  common abbreviation `"pdr"` — so `"pdr room"` skips the bathroom branch
  entirely and falls through to the generic `"room"` match instead.
- `"covliiog toica"` (true: covered porch, `others`) → misclassified
  `bathrooms` via `"toi"`. Same case already identified in the 15-image
  report; still present, not new.

**Where the fix helped and didn't hold up at the larger N.** Kitchens and
rooms recall both improved substantially (0.70→0.81, 0.62→0.68); halls
recall actually *dropped slightly* (0.58→0.55) rather than staying flat
as the 15-image report found — consistent with that report's own
explanation (`"hall"`/`"liv"`/`"great"` are longer, pickier substrings
that residual OCR noise keeps landing on) rather than contradicting it,
just a more precise measurement of the same effect.

**Precision costs are also confirmed at scale.** Bathrooms precision
1.00→0.98 and rooms precision 0.77→0.75 both dropped for the same reason
identified in the 15-image report: more legible text now reaches
`classify_room_label` instead of failing silently into `others`, which
also means more chances for the substring-logic bugs (documented and
newly found above) to fire.

## Original 15-image subset (untouched holdout, no longer the headline)

Kept for transparency about what the original report claimed and how it
compares. Ground truth, split, and crops unchanged from the original
measurement (`eval/split.json`, `eval/detections.csv`,
`eval/detections_excluded_unreadable.csv`, `eval/crops/`).

### Before

| true \ pred | kitchens | bathrooms | halls | rooms | others | support |
|---|---|---|---|---|---|---|
| **kitchens** | 7 | 0 | 0 | 0 | 10 | 17 |
| **bathrooms** | 0 | 7 | 0 | 0 | 25 | 32 |
| **halls** | 0 | 0 | 4 | 0 | 12 | 16 |
| **rooms** | 0 | 0 | 0 | 14 | 22 | 36 |
| **others** | 0 | 0 | 0 | 1 | 170 | 171 |

Accuracy: 202/272 = 0.74 (N=272).

### After

| true \ pred | kitchens | bathrooms | halls | rooms | others | support |
|---|---|---|---|---|---|---|
| **kitchens** | 11 | 0 | 0 | 0 | 6 | 17 |
| **bathrooms** | 0 | 9 | 0 | 0 | 23 | 32 |
| **halls** | 0 | 0 | 4 | 3 | 9 | 16 |
| **rooms** | 0 | 0 | 0 | 19 | 17 | 36 |
| **others** | 0 | 1 | 0 | 2 | 168 | 171 |

Accuracy: 211/272 = 0.78 (N=272). Empty-OCR-rate: 128/272 (47%) → 30/272
(11%) — consistent in direction and magnitude with the full-set primary
result above (33.6%→10.7%), which is expected since this subset is
contained within the full set, not an independent check of it.

## Recommendation (still not implemented — for sign-off)

The full-set numbers make the original recommendation's case *more*
solid, not less: 100% of the classifier's errors in the original
15-image sample traced to the OCR step, and the full-set investigation
above adds two more confirmed substring-logic bugs (`"nooki"`/parking-
shaped, `"pdr"` unrecognized) that only had room to surface once OCR
recall stopped hiding them. The recommendation is unchanged: the OCR
crop fix (implemented) was the right first move, and the substring rules
are now the clearest next lever — specifically the newly-confirmed
`"ki"`-anywhere and unrecognized-abbreviation gaps, plus the
already-documented `"br"`/library-breakfast and `"room"`-inside-"mud
room" cases. Still not implemented here, per "don't touch until you
decide."

## Reproduce the post-fix 15-image numbers specifically

```bash
docker compose build api   # picks up prepare_ocr_crop
docker run --rm \
  -v "$(pwd)/scripts:/app/scripts:ro" -v "$(pwd)/eval:/app/eval" -w /app \
  smartsense-project-api:latest python scripts/eval_floorplan_classifier.py extract \
    --images-dir /app/data/images --out /app/eval --eval-fraction 0.2 --seed 42
python scripts/eval_floorplan_classifier.py score --detections eval/detections.csv --out eval/REPORT.md
```
