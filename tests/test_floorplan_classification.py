"""
Tests for src/floorplan.classify_room_label() (no YOLO or EasyOCR).

They record the current substring behaviour, including known false
positives: "ki" matches "parking", "br" matches "library" and "breakfast",
and "bathroom" is classified correctly only because the bathroom check runs
before the room check. Fixing the classifier will require updating these
tests.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.floorplan import classify_room_label  # noqa: E402

# --- Kitchens: "ki" substring ---


def test_kitchen_matches():
    assert classify_room_label("kitchen") == "kitchens"


def test_kitchen_false_positive_parking():
    # Known bug: "parking" contains "ki", so it's counted as a kitchen.
    assert classify_room_label("parking") == "kitchens"


def test_kitchen_branch_wins_even_with_bath_also_present():
    # "ki" is checked before any bathroom keyword, so a label matching
    # both takes the kitchen branch.
    assert classify_room_label("kitchenette bath") == "kitchens"


# --- Bathrooms: "bath" / "wc" / "wash" / "toi" / "powder" ---


def test_bathroom_bath_keyword():
    assert classify_room_label("bath") == "bathrooms"


def test_bathroom_wc_keyword():
    assert classify_room_label("wc") == "bathrooms"


def test_bathroom_wash_keyword():
    assert classify_room_label("washroom") == "bathrooms"


def test_bathroom_toilet_keyword():
    assert classify_room_label("toilet") == "bathrooms"


def test_bathroom_powder_keyword():
    assert classify_room_label("powder room") == "bathrooms"


def test_bathroom_wins_over_room_due_to_branch_order():
    # "bathroom" contains both "bath" and "room"; it's a bathroom only
    # because that branch runs first. Guards against reordering.
    assert classify_room_label("bathroom") == "bathrooms"
    assert classify_room_label("master bath") == "bathrooms"


# --- Halls: "hall" / "liv" / "great" ---


def test_hall_keyword():
    assert classify_room_label("hallway") == "halls"


def test_hall_liv_keyword():
    assert classify_room_label("living") == "halls"


def test_hall_wins_over_room_for_living_room():
    # "living room" contains "liv" (hall) and "room"; the hall branch runs
    # first.
    assert classify_room_label("living room") == "halls"


def test_hall_great_keyword_wins_over_room():
    # "great room" contains both "great" (hall) and "room" (bedroom); the
    # hall branch wins for the same reason.
    assert classify_room_label("great room") == "halls"


def test_hall_entrance_hall():
    assert classify_room_label("entrance hall") == "halls"


# --- Rooms (bedrooms): "bed" / "room" / "br" ---


def test_bedroom_bed_keyword():
    assert classify_room_label("bedroom") == "rooms"


def test_bedroom_generic_room_keyword():
    assert classify_room_label("room") == "rooms"


def test_bedroom_br_keyword():
    assert classify_room_label("master br") == "rooms"
    assert classify_room_label("br") == "rooms"


def test_br_false_positive_library():
    # Known bug: "library" contains "br", so it's counted as a bedroom.
    assert classify_room_label("library") == "rooms"


def test_br_false_positive_breakfast():
    # Known bug: "breakfast" contains "br", so it's counted as a bedroom.
    assert classify_room_label("breakfast") == "rooms"
    assert classify_room_label("breakfast nook") == "rooms"


# --- Others: no keyword matches ---


def test_others_fallback_for_unrelated_labels():
    for label in ("storage", "office", "den", "garage", "closet", "pantry", "entry", "terrace", "veranda"):
        assert classify_room_label(label) == "others", f"expected 'others' for {label!r}"


def test_others_for_empty_string():
    assert classify_room_label("") == "others"


def test_uppercase_input_is_not_matched():
    # The function expects lowercase input (parse_floorplan lowercases it),
    # so uppercase text falls through to "others".
    assert classify_room_label("KITCHEN") == "others"


CASES = [
    test_kitchen_matches,
    test_kitchen_false_positive_parking,
    test_kitchen_branch_wins_even_with_bath_also_present,
    test_bathroom_bath_keyword,
    test_bathroom_wc_keyword,
    test_bathroom_wash_keyword,
    test_bathroom_toilet_keyword,
    test_bathroom_powder_keyword,
    test_bathroom_wins_over_room_due_to_branch_order,
    test_hall_keyword,
    test_hall_liv_keyword,
    test_hall_wins_over_room_for_living_room,
    test_hall_great_keyword_wins_over_room,
    test_hall_entrance_hall,
    test_bedroom_bed_keyword,
    test_bedroom_generic_room_keyword,
    test_bedroom_br_keyword,
    test_br_false_positive_library,
    test_br_false_positive_breakfast,
    test_others_fallback_for_unrelated_labels,
    test_others_for_empty_string,
    test_uppercase_input_is_not_matched,
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
