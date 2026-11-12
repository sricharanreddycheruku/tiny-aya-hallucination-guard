"""Compare automatic correctness with completed human review annotations."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from followup import read_jsonl
from scoring import is_wrong


def summarize_review(annotations, answers):
    by_language = defaultdict(lambda: {"fact_reviewed": 0, "old_agrees": 0, "new_agrees": 0,
                                      "support_reviewed": 0, "supported": 0})
    seen = set()
    for annotation in annotations:
        sample_id = annotation["id"]
        if sample_id in seen or sample_id not in answers:
            raise ValueError(f"Duplicate or unknown review ID: {sample_id}")
        seen.add(sample_id)
        row = answers[sample_id]
        stats = by_language[row["language"]]
        fact = annotation["factually_correct"].strip().lower()
        support = annotation["supported_by_passage"].strip().lower()
        if fact not in {"", "yes", "no", "unclear"} or support not in {"", "yes", "no", "unclear"}:
            raise ValueError(f"Use yes, no or unclear for {sample_id}.")
        if fact in {"yes", "no"}:
            human_wrong = fact == "no"
            revised_wrong = is_wrong(row["answer"], row["golds"], row["language"])
            stats["fact_reviewed"] += 1
            stats["old_agrees"] += int(row["wrong"] == human_wrong)
            stats["new_agrees"] += int(revised_wrong == human_wrong)
        if support in {"yes", "no"}:
            stats["support_reviewed"] += 1
            stats["supported"] += int(support == "yes")
    return dict(sorted(by_language.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_csv", type=Path)
    parser.add_argument("answers_jsonl", type=Path)
    args = parser.parse_args()
    answers = {row["id"]: row for row in read_jsonl(args.answers_jsonl)}
    with args.review_csv.open(newline="", encoding="utf-8-sig") as handle:
        result = summarize_review(csv.DictReader(handle), answers)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
