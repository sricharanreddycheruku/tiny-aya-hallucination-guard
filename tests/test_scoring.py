import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scoring import exact_match, is_wrong, normalize, token_f1


def test_english_articles_and_punctuation():
    assert exact_match("The Eiffel Tower.", ["eiffel tower"], "en")


def test_articles_kept_outside_english():
    assert normalize("a b", "sw") == "a b"


def test_telugu_vowel_signs_are_preserved():
    assert normalize("కాలం") != normalize("కలం")
    assert not exact_match("కాలం", ["కలం"], "te")


def test_telugu_match_ignores_punctuation():
    assert exact_match("హైదరాబాద్.", ["హైదరాబాద్"], "te")


def test_devanagari_danda_is_removed():
    assert exact_match("दिल्ली।", ["दिल्ली"], "hi")


def test_f1_takes_best_gold_and_handles_empty():
    assert token_f1("new delhi india", ["delhi", "new delhi india"], "en") == 1.0
    assert token_f1("", ["delhi"], "en") == 0.0


def test_is_wrong():
    assert is_wrong("mumbai", ["new delhi"], "en")
    assert not is_wrong("new delhi", ["delhi"], "en")


def test_shared_unit_does_not_rescue_a_wrong_number():
    assert is_wrong("65 కి.మీ.", ["45 కి.మీ"], "te")
    assert is_wrong("300 km", ["42 km"], "en")
    assert not is_wrong("45 కి.మీ.", ["45 కి.మీ"], "te")
    assert not is_wrong("1773", ["December 16, 1773"], "en")


def test_suffix_attached_to_a_number_is_separated():
    assert not is_wrong("1971లో", ["1971"], "te")
    assert is_wrong("1992లో", ["1971"], "te")


def test_date_components_and_abbreviations():
    gold = ["December 16, 1773"]
    assert is_wrong("December 17, 1773", gold, "en")
    assert is_wrong("November 16, 1773", gold, "en")
    assert not is_wrong("1773", gold, "en")
    assert not is_wrong("23 Machi 1942", ["1942"], "sw")
    assert not is_wrong("Mid-1970s", ["1970s"], "en")
    assert exact_match("2026-11-12", ["2026 11 12"])
    assert is_wrong("১৯৪২ সালের ৯ জানুয়ারি", ["১৯৪২ সালের ৮ জানুয়ারি"], "bn")


def test_decimal_quantities_and_units():
    assert is_wrong("2 mg", ["2.5 mg"], "en")
    assert is_wrong("2.5 g", ["2.5 mg"], "en")
    assert is_wrong("−2.5 mg", ["2.5 mg"], "en")
    assert exact_match("২.৫০", ["2.5"], "bn")
    assert exact_match("౨.౫౦ కి.మీ.", ["2.5 కి.మీ"], "te")
    assert exact_match("1,000.50", ["1000.5"], "en")
    assert exact_match("2,7", ["2.7"], "sw")
    assert not is_wrong("98.8 TWh.", ["98.8 terawatt-hours"], "en")
    assert not is_wrong("mlima Teide ina urefu wa mita 3,718.", ["mita 3,718"], "sw")
