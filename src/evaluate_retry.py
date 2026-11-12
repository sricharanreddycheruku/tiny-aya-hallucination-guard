"""Compare the frozen guard with one optional retry on the saved health questions."""
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from followup import read_jsonl, slices, summarize
from scoring import is_wrong

ROOT = Path(__file__).resolve().parents[1]


def score_retry(rows, pipeline, rule):
    first_delivered, retry_delivered = [], []
    first_wrong, retry_wrong = [], []
    base_seconds, retry_seconds = [], []
    attempts = []
    for row in rows:
        started = perf_counter()
        result = pipeline.respond(
            row["question"], row["language"], rule=rule, retry=True,
            excluded_topic=row["title"] if not row["covered"] else None,
        )
        elapsed = perf_counter() - started
        first = result.get("first_attempt", result)
        duration = result.get("retry_seconds", 0.0)
        attempts.append(result.get("retrieval_attempts", 1))
        base_seconds.append(elapsed - duration)
        retry_seconds.append(duration)
        first_delivered.append(first["delivered"])
        retry_delivered.append(result["delivered"])
        first_wrong.append(is_wrong(first["answer"], row["golds"], row["language"]))
        retry_wrong.append(is_wrong(result["answer"], row["golds"], row["language"]))

    first_delivered, retry_delivered, retry_wrong = map(np.asarray, (first_delivered, retry_delivered, retry_wrong))
    recovered = ~first_delivered & retry_delivered
    groups = slices(rows)
    # answer_in_context describes the previously saved passage, not the retry.
    groups = {key: mask for key, mask in groups.items() if not key.startswith("gold_string")}
    return {
        "rule": rule,
        "labels": "revised automatic scorer",
        "single_pass": summarize(first_delivered, np.array(first_wrong), groups),
        "with_retry": summarize(retry_delivered, retry_wrong, groups),
        "retried": attempts.count(2),
        "correct_answers_recovered": int((recovered & ~retry_wrong).sum()),
        "additional_wrong_delivered": int((recovered & retry_wrong).sum()),
        "mean_single_pass_seconds": float(np.mean(base_seconds)),
        "mean_added_seconds": float(np.mean(retry_seconds)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "data" / "health_big.jsonl")
    parser.add_argument("--assets", type=Path, default=ROOT / "space_build")
    parser.add_argument("--rule", choices=("joint", "selective"), default="joint")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "followup" / "health_retry.json")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "space"))
    from pipeline import Pipeline

    rows = read_jsonl(args.questions)
    if not rows:
        parser.error("The health question file is empty.")
    pipeline = Pipeline(args.assets)
    # Warm up generation and the encoder before timing the paired run.
    pipeline.respond(rows[0]["question"], rows[0]["language"])
    result = score_retry(rows, pipeline, args.rule)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
