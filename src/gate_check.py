"""How well the CuratorKIT judge separates grounded answers from planted wrong ones.

The judge writes its key with a leading space, so the gate reads the score through
its raw-text fallback. The first number in the reply is that score, so scores are valid.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from curatorkit.gates.hallucination import HallucinationGate

from curate import LocalLLM

THRESHOLDS = (0.7, 0.8, 0.9, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/health_questions.jsonl")
    parser.add_argument("--judge", default="CohereLabs/tiny-aya-global")
    parser.add_argument("--limit", type=int, default=60, help="questions per language")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="../runs/gate_check.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    grouped = defaultdict(list)
    for line in open(args.data, encoding="utf-8"):
        row = json.loads(line)
        grouped[row["language"]].append(row)

    gate = HallucinationGate(llm=LocalLLM(args.judge, max_tokens=96))
    report = {}
    for language, rows in grouped.items():
        picked = rng.sample(rows, min(args.limit, len(rows)))
        true_scores, planted_scores = [], []
        for row in picked:
            others = [r["golds"][0] for r in rows if r["golds"][0] not in row["source_text"]]
            true_scores.append(gate._evaluate_grounding(row["source_text"], row["question"], row["golds"][0])[0])
            planted_scores.append(gate._evaluate_grounding(row["source_text"], row["question"], rng.choice(others))[0])
        report[language] = {
            "n": len(picked), "true_scores": true_scores, "planted_scores": planted_scores,
            "pass_rates": {
                str(t): {
                    "true": sum(s >= t for s in true_scores) / len(picked),
                    "planted": sum(s >= t for s in planted_scores) / len(picked),
                } for t in THRESHOLDS
            },
        }
        print(language, report[language]["pass_rates"], flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
