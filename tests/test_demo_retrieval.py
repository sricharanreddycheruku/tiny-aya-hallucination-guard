import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "space"))
from pipeline import Pipeline


def demo_pipeline():
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.device = "cpu"
    pipeline.corpus = [
        {"text": "first", "topic": "held-out", "language": "en"},
        {"text": "first", "topic": "held-out", "language": "en"},
        {"text": "second", "topic": "other", "language": "en"},
        {"text": "higher scoring foreign passage", "topic": "other", "language": "sw"},
    ]
    pipeline.languages = np.array([row["language"] for row in pipeline.corpus])
    pipeline.topics = np.array([row["topic"] for row in pipeline.corpus])
    pipeline.vectors = torch.tensor([[0.9], [0.9], [0.8], [1.0]])
    pipeline.embed = lambda text, prefix: torch.tensor([1.0])
    return pipeline


def test_rank_passages_filters_language_topic_and_duplicate_text():
    pipeline = demo_pipeline()
    query = torch.tensor([1.0])
    ranked = pipeline.rank_passages(query, "en", limit=2)
    assert [passage["text"] for passage, _ in ranked] == ["first", "second"]
    eligible = pipeline.rank_passages(query, "en", limit=2, excluded_topic="held-out")
    assert [passage["text"] for passage, _ in eligible] == ["second"]


def test_retry_only_after_abstention_and_second_answer_is_rescored():
    pipeline = demo_pipeline()
    calls = []

    def score(question, passage, similarity, rule):
        calls.append((passage["text"], rule))
        return {"answer": passage["text"], "delivered": passage["text"] == "second"}

    pipeline.score_passage = score
    assert not pipeline.respond("question", "en")["delivered"]
    assert calls == [("first", "joint")]
    calls.clear()
    result = pipeline.respond("question", "en", rule="selective", retry=True)
    assert result["first_attempt"]["answer"] == "first"
    assert result["answer"] == "second" and result["delivered"]
    assert calls == [("first", "selective"), ("second", "selective")]
    calls.clear()
    result = pipeline.respond("question", "en", excluded_topic="held-out", retry=True)
    assert result["delivered"]
    assert calls == [("second", "joint")]
