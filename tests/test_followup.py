import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate_retry import score_retry
from followup import evaluate, pilot_split, write_review
from review_agreement import summarize_review


def records(n):
    return [
        {"id": str(i), "language": "en", "title": f"article-{i // 2}", "question": f"Question {i}",
         "answer": "2.5 mg", "golds": ["2.5 mg"], "wrong": i >= n // 2,
         "covered": i % 2 == 0, "answer_in_context": i % 3 == 0}
        for i in range(n)
    ]


def test_majority_thresholds_use_calibration_scores_and_report_groups(tmp_path):
    cal = records(20)
    test = records(10)
    cal_scores = {"guard": np.arange(20) / 20, "no_activation": np.arange(20) / 40}
    scores = {"guard": np.arange(10) / 10, "no_activation": np.arange(10) / 20}
    report, decisions = evaluate(cal, cal_scores, test, scores)
    guard = report["majority"]["0.55"]["guard"]
    assert guard["threshold"] == 0.5
    assert guard["test"]["all"]["answered"] == 6
    assert guard["test"]["all"]["wrong_delivered"] == 1
    assert guard["test"]["source_article_excluded"]["n"] == 5
    assert guard["test"]["gold_string_present"]["n"] == 4
    assert np.all(pilot_split(cal)[::2] == pilot_split(cal)[1::2])

    review = tmp_path / "review.csv"
    write_review(review, test, decisions[0.60, "guard"], {r["id"]: "passage" for r in test}, per_group=2)
    with review.open(newline="", encoding="utf-8-sig") as handle:
        sample = list(csv.DictReader(handle))
    assert len(sample) == 4
    assert "wrong" not in sample[0]
    assert all(not r["factually_correct"] and not r["supported_by_passage"] for r in sample)


def test_retry_report_separates_correct_recoveries_from_errors():
    rows = records(3)

    class FakePipeline:
        def respond(self, question, language, rule, retry, excluded_topic):
            i = int(question.rsplit(" ", 1)[1])
            assert retry
            assert excluded_topic == (rows[i]["title"] if i % 2 else None)
            first = {"answer": "2 mg", "delivered": i == 0}
            if first["delivered"]:
                return first
            return {"answer": "2.5 mg" if i == 1 else "2 mg", "delivered": True,
                    "first_attempt": first, "retrieval_attempts": 2}

    report = score_retry(rows, FakePipeline(), "joint")
    assert report["single_pass"]["all"]["wrong_delivered"] == 1
    assert report["with_retry"]["all"]["answered"] == 3
    assert report["correct_answers_recovered"] == 1
    assert report["additional_wrong_delivered"] == 1


def test_human_review_compares_old_and_revised_labels():
    answer = dict(records(1)[0], answer="2 mg", wrong=False)
    review = {"id": "0", "factually_correct": "no", "supported_by_passage": "unclear"}
    assert summarize_review([review], {"0": answer}) == {
        "en": {"fact_reviewed": 1, "old_agrees": 0, "new_agrees": 1,
               "support_reviewed": 0, "supported": 0}
    }
