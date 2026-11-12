"""Ground-truth answer scoring for short-answer QA in any script.

Only punctuation and symbols are removed. Combining marks stay, because in
Indic scripts they carry vowels, so stripping them would merge different words.
"""
import re
import unicodedata
from collections import Counter
from decimal import Decimal

ENGLISH_ARTICLES = re.compile(r"\b(a|an|the)\b")
DIGIT_BOUNDARY = re.compile(r"(?<=\d)(?=[^\d\s])|(?<=[^\d\s])(?=\d)")
NUMBER = re.compile(r"(?<!\d)(?:(?<!\w)[+−-])?(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:[.٫]\d+)?")
COMMA_DECIMAL = re.compile(r"(?<!\d)(\d+),(\d{1,2})(?!\d)")
MONTHS = set("january february march april may june july august september october november december".split())
UNIT_GROUPS = (
    ("mg", "milligram", "milligrams"), ("g", "gram", "grams"), ("kg", "kilogram", "kilograms"),
    ("km", "kilometre", "kilometer", "kilometres", "kilometers"),
    ("mi", "mile", "miles"), ("m", "metre", "meter", "metres", "meters"),
    ("ml", "millilitre", "milliliter", "millilitres", "milliliters"),
    ("l", "litre", "liter", "litres", "liters"), ("%", "percent"),
)
UNITS = {form: group[0] for group in UNIT_GROUPS for form in group}


def normalize(text, language=None):
    text = unicodedata.normalize("NFC", text).casefold()
    text = "".join(" " if unicodedata.category(c)[0] in "PSZ" else c for c in text)
    text = DIGIT_BOUNDARY.sub(" ", text)
    if language == "en":
        text = ENGLISH_ARTICLES.sub(" ", text)
    return " ".join(text.split())


def answer_tokens(text, language=None):
    """Keep decimals and signed quantities intact for answer scoring."""
    if language == "sw":
        text = COMMA_DECIMAL.sub(r"\1.\2", text)
    tokens = []
    start = 0
    for match in NUMBER.finditer(text):
        # The lookbehind leaves a hyphen between digits as a separator.
        raw = match[0].replace("−", "-").replace("٫", ".").replace(",", "")
        tokens.extend(normalize(text[start:match.start()], language).split())
        value = format(Decimal(raw), "f")
        tokens.append(value.rstrip("0").rstrip(".") if "." in value else value)
        start = match.end()
        if text[start:start + 1] == "%":
            tokens.append("%")
            start += 1
    tokens.extend(normalize(text[start:], language).split())
    return tokens


def exact_match(prediction, golds, language=None):
    pred = answer_tokens(prediction, language)
    return any(pred == answer_tokens(g, language) for g in golds)


def overlap_f1(pred, ref):
    overlap = sum((Counter(pred) & Counter(ref)).values())
    return 2 * overlap / (len(pred) + len(ref)) if overlap else 0.0


def number_tokens(tokens):
    return {token for token in tokens if NUMBER.fullmatch(token)}


def token_f1(prediction, golds, language=None):
    pred = answer_tokens(prediction, language)
    predicted = number_tokens(pred)
    best = 0.0
    for gold in golds:
        ref = answer_tokens(gold, language)
        score = overlap_f1(pred, ref)
        if predicted and predicted == number_tokens(ref):
            # Preserve credit given by the old tokenizer when the quantities
            # really agree and canonicalization reduced the shared token count.
            score = max(score, overlap_f1(normalize(prediction, language).split(), normalize(gold, language).split()))
        best = max(best, score)
    return best


def numbers(text, language=None):
    return number_tokens(answer_tokens(text, language))


def is_wrong(prediction, golds, language=None, f1_cutoff=0.5):
    """Correct needs enough token overlap with a gold answer and no number conflict.

    An abbreviated numeric answer can omit parts of the reference, but any
    numbers it gives must agree. Explicit conflicting units or months also fail.
    """
    predicted = numbers(prediction, language)
    for gold in golds:
        expected = numbers(gold, language)
        if predicted and expected and not (predicted <= expected or expected <= predicted):
            continue
        if predicted and expected and language == "en":
            pred_tokens = set(answer_tokens(prediction, language))
            ref_tokens = set(answer_tokens(gold, language))
            pred_units = {UNITS[t] for t in pred_tokens & UNITS.keys()}
            ref_units = {UNITS[t] for t in ref_tokens & UNITS.keys()}
            if pred_units and ref_units and not pred_units & ref_units:
                continue
            pred_months, ref_months = pred_tokens & MONTHS, ref_tokens & MONTHS
            if pred_months and ref_months and pred_months != ref_months:
                continue
        if token_f1(prediction, [gold], language) >= f1_cutoff:
            return False
    return True
