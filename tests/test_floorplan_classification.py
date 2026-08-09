"""
Tests for src/floorplan.py's classify_room_label() -- a pure function, no
YOLO or EasyOCR involved. Exercises the current substring-cascade behavior
exhaustively, including the known false positives flagged in
smartsense-fix-list.md's Tier 4 section: "ki" matches "parking", "br"
matches "library" and "breakfast", and "room" matches "bathroom" but only
resolves correctly because the bathroom branch runs first. These tests
pin the *current* (imperfect) behavior deliberately -- they are not meant
to pass judgment on it. When Tier 4 replaces or fixes the classifier,
these assertions are expected to need deliberate updates, not silent
breakage.
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
    # Documented bug: "parking" contains "ki" (par-KI-ng), so a parking
    # label gets counted as a kitchen. See fix-list Tier 4.
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
    # Load-bearing ordering, called out explicitly in the fix list:
    # "bathroom" contains both "bath" and "room". It only resolves to
    # "bathrooms" because the bathroom elif runs before the room elif.
    # If these branches were ever reordered, this would silently flip to
    # "rooms" -- that's exactly what this test guards against.
    assert classify_room_label("bathroom") == "bathrooms"
    assert classify_room_label("master bath") == "bathrooms"


# --- Halls: "hall" / "liv" / "great" ---

def test_hall_keyword():
    assert classify_room_label("hallway") == "halls"


def test_hall_liv_keyword():
    assert classify_room_label("living") == "halls"


def test_hall_wins_over_room_for_living_room():
    # "living room" contains both "liv" (hall) and "room" (bedroom); the
    # hall branch runs first, so it wins -- same ordering dependency as
    # the bathroom case above, just less obviously a "bug".
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
    # Documented bug: "library" contains "br" (li-BR-ary), so it's
    # counted as a bedroom. See fix-list Tier 4.
    assert classify_room_label("library") == "rooms"


def test_br_false_positive_breakfast():
    # Documented bug: "breakfast" contains "br" (BR-eakfast), so it's
    # counted as a bedroom too.
    assert classify_room_label("breakfast") == "rooms"
    assert classify_room_label("breakfast nook") == "rooms"


# --- Others: no keyword matches ---

def test_others_fallback_for_unrelated_labels():
    for label in ("storage", "office", "den", "garage", "closet", "pantry", "entry", "terrace", "veranda"):
        assert classify_room_label(label) == "others", f"expected 'others' for {label!r}"


def test_others_for_empty_string():
    assert classify_room_label("") == "others"


def test_uppercase_input_is_not_matched():
    # classify_room_label does no case-folding itself -- it's documented
    # (and relied on by parse_floorplan) to receive already-lowercased
    # text. Uppercase "KITCHEN" doesn't contain lowercase "ki", so it
    # falls through to "others". This pins that contract: if someone
    # removes the .lower() call in parse_floorplan, classification
    # silently degrades to "others" for everything rather than raising.
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
