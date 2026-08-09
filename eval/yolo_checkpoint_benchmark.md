# YOLO checkpoint benchmark: best_1000.pt vs best_300.pt

Reproduce with: `git show aea4a40^:src/best_300.pt > src/best_300.pt && python scripts/benchmark_yolo_checkpoints.py`

N = 16 hand-labeled images (of 73 in `data/images/`). Error = |predicted room_name box count - hand-counted true count|, per image. Ground truth methodology and spot-check in the script's docstring.

## Per-image results

| image | GT | best_1000 | best_300 | err (1000) | err (300) | winner |
|---|---|---|---|---|---|---|
| 0_20_jpg.rf.55d03cd6dd66cf512099eb67e527175c.jpg | 14 | 16 | 16 | 2 | 2 | tie |
| 0_26_jpg.rf.22d020e2780fff58cefb479d8fd41058.jpg | 12 | 12 | 14 | 0 | 2 | best_1000 |
| 0_28_jpg.rf.bda89f2956b68f4dafae375c7ee2af77.jpg | 11 | 13 | 14 | 2 | 3 | best_1000 |
| 0_30_jpg.rf.bcf0ea49d5693cfb9bf9fd3efb73181b.jpg | 18 | 25 | 30 | 7 | 12 | best_1000 |
| 10_21_jpg.rf.cdc2b334287b64744b80fa37e5125068.jpg | 21 | 29 | 23 | 8 | 2 | best_300 |
| 10_26_jpg.rf.e5d27a70ba2a91011c456723cda40b4a.jpg | 15 | 16 | 14 | 1 | 1 | tie |
| 10_29_jpg.rf.0c5067935034641091bb2290b5d245ee.jpg | 6 | 6 | 6 | 0 | 0 | tie |
| 13_18_jpg.rf.ec11791b4d9cc45c6a4809fd27fd1ca1.jpg | 9 | 8 | 7 | 1 | 2 | best_1000 |
| 15_25_jpg.rf.cbfd098a4c0bc0edd2d9e9b1620b4bb5.jpg | 19 | 21 | 23 | 2 | 4 | best_1000 |
| 16_11_jpg.rf.e0a76b0fba37d1b966021491bf85cd68.jpg | 24 | 27 | 31 | 3 | 7 | best_1000 |
| 16_15_jpg.rf.dab1858a93afa721d4ebbdb824c68898.jpg | 17 | 16 | 18 | 1 | 1 | tie |
| 16_24_jpg.rf.cacc0b0120946b0e3f572939c3ff3206.jpg | 10 | 7 | 7 | 3 | 3 | tie |
| 17_17_jpg.rf.7623bd0247831a1ec9a129d372486d39.jpg | 16 | 15 | 15 | 1 | 1 | tie |
| 28_18_jpg.rf.7e2a025d6ea364275d34bcec17eff91d.jpg | 14 | 11 | 9 | 3 | 5 | best_1000 |
| 66_6_jpg.rf.2bbd0923857fe025ca96d559913b98ef.jpg | 20 | 25 | 24 | 5 | 4 | best_300 |
| 72_8_jpg.rf.bc78aa4d12e97a36a4cb199d0b7456aa.jpg | 16 | 11 | 11 | 5 | 5 | tie |

## Summary

**Wins / losses / ties (out of 16):** best_1000 wins 7, best_300 wins 2, ties 7.

**Mean absolute error per image:** best_1000 = 2.750, best_300 = 3.375 (18.5% lower for best_1000).

best_1000 wins the plurality of decisive comparisons (7 of 9 non-tied images) and has the lower MAE -- consistent with the original decision to keep it over best_300.
