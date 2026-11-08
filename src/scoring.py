"""Ground-truth answer scoring for short-answer QA in any script.

Only punctuation and symbols are removed. Combining marks stay, because in
Indic scripts they carry vowels, so stripping them would merge different words.
"""
import re
import unicodedata
from collections import Counter

ENGLISH_ARTICLES = re.compile(r"\b(a|an|the)\b")
DIGIT_BOUNDARY = re.compile(r"(?<=\d)(?=[^\d\s])|(?<=[^\d\s])(?=\d)")


def normalize(text, language=None):
    text = unicodedata.normalize("NFC", text).casefold()
    text = "".join(" " if unicodedata.category(c)[0] in "PSZ" else c for c in text)
    text = DIGIT_BOUNDARY.sub(" ", text)
    if language == "en":
        text = ENGLISH_ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction, golds, language=None):
    pred = normalize(prediction, language)
    return any(pred == normalize(g, language) for g in golds)


def token_f1(prediction, golds, language=None):
    pred = normalize(prediction, language).split()
    best = 0.0
    for gold in golds:
        ref = normalize(gold, language).split()
        overlap = sum((Counter(pred) & Counter(ref)).values())
        if overlap:
            precision, recall = overlap / len(pred), overlap / len(ref)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def numbers(text, language=None):
    return {token for token in normalize(text, language).split() if any(c.isdigit() for c in token)}


def is_wrong(prediction, golds, language=None, f1_cutoff=0.5):
    """Correct needs enough token overlap with a gold answer and no number conflict.

    Shared units must not rescue a wrong quantity, so when both sides contain
    numbers at least one has to match.
    """
    predicted = numbers(prediction, language)
    for gold in golds:
        expected = numbers(gold, language)
        if predicted and expected and not predicted & expected:
            continue
        if token_f1(prediction, [gold], language) >= f1_cutoff:
            return False
    return True
