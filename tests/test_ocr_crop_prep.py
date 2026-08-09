"""
Tests for src/floorplan.py's prepare_ocr_crop() -- the padding/upscaling
step added after eval/REPORT.md found that undersized crops (not the
substring rules) accounted for 100% of the classifier's real-category
misclassifications in its held-out measurement. Pure PIL/numpy, no YOLO
or EasyOCR involved.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PIL import Image  # noqa: E402

from src.floorplan import prepare_ocr_crop  # noqa: E402

MIN_HEIGHT = 48
MAX_SCALE = 6


def _blank_image(w, h):
    return Image.new("RGB", (w, h), "white")


def test_undersized_crop_is_upscaled_to_at_least_min_height():
    img = _blank_image(200, 200)
    crop = prepare_ocr_crop(img, 10, 10, 40, 20)  # 30x10 box, well under MIN_HEIGHT
    assert crop.shape[0] >= MIN_HEIGHT


def test_crop_already_tall_enough_is_not_upscaled():
    img = _blank_image(200, 200)
    # padded height comfortably clears MIN_HEIGHT before any scaling
    crop = prepare_ocr_crop(img, 20, 20, 120, 80)  # 100x60 box
    padded_h = 60 + 2 * max(2, int(60 * 0.15))
    assert padded_h >= MIN_HEIGHT
    assert crop.shape[0] == padded_h


def test_padding_expands_beyond_the_raw_box():
    img = _blank_image(200, 200)
    x1, y1, x2, y2 = 50, 50, 150, 90  # 100x40 box, padded height still >= MIN_HEIGHT
    crop = prepare_ocr_crop(img, x1, y1, x2, y2)
    assert crop.shape[1] > (x2 - x1)  # width grew from padding
    assert crop.shape[0] > (y2 - y1)  # height grew from padding


def test_upscale_is_capped_at_max_scale():
    img = _blank_image(500, 500)
    crop = prepare_ocr_crop(img, 10, 10, 15, 12)  # a near-degenerate 5x2 box
    padded_h = 2 + 2 * max(2, int(2 * 0.15))
    assert crop.shape[0] <= padded_h * MAX_SCALE


def test_box_at_image_edge_does_not_error_or_go_out_of_bounds():
    img = _blank_image(50, 50)
    crop = prepare_ocr_crop(img, 0, 0, 10, 5)  # touches the top-left corner
    assert crop.shape[0] > 0 and crop.shape[1] > 0

    crop2 = prepare_ocr_crop(img, 40, 45, 50, 50)  # touches the bottom-right corner
    assert crop2.shape[0] > 0 and crop2.shape[1] > 0


def test_aspect_ratio_is_preserved_when_upscaling():
    img = _blank_image(200, 200)
    x1, y1, x2, y2 = 10, 10, 60, 18  # 50x8 box
    pad_x = max(2, int(50 * 0.15))
    pad_y = max(2, int(8 * 0.15))
    padded_w, padded_h = (x2 - x1) + 2 * pad_x, (y2 - y1) + 2 * pad_y
    crop = prepare_ocr_crop(img, x1, y1, x2, y2)
    expected_ratio = padded_w / padded_h
    actual_ratio = crop.shape[1] / crop.shape[0]
    assert abs(actual_ratio - expected_ratio) < 0.05


CASES = [
    test_undersized_crop_is_upscaled_to_at_least_min_height,
    test_crop_already_tall_enough_is_not_upscaled,
    test_padding_expands_beyond_the_raw_box,
    test_upscale_is_capped_at_max_scale,
    test_box_at_image_edge_does_not_error_or_go_out_of_bounds,
    test_aspect_ratio_is_preserved_when_upscaling,
]


def main() -> int:
    failures = 0
    for case in CASES:
        try:
            case()
        except AssertionError as e:
            failures += 1
            print(f"FAIL {case.__name__}: {e}")
        else:
            print(f"PASS {case.__name__}")
    if failures:
        print(f"\n{failures}/{len(CASES)} tests failed")
        return 1
    print(f"\nAll {len(CASES)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
