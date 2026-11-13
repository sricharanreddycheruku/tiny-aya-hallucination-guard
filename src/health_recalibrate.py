"""Evaluate health-specific thresholds on topics held out from calibration."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from calibration import conformal_threshold, per_language_thresholds
from followup import read_jsonl, revised_labels, summarize
from guard import Guard, LAYER


def calibration_topics(topics):
    unique = sorted(set(topics), key=lambda topic: hashlib.sha256(
        f"health-recalibration-v1/{topic}".encode()).digest())
    if len(unique) < 2:
        raise ValueError("At least two distinct health topics are required.")
    return set(unique[:max(1, int(0.4 * len(unique)))])


def compare(rows, scores, frozen, alpha, seed=0):
    topics = np.array([row["title"] for row in rows])
    languages = np.array([row["language"] for row in rows])
    wrong = np.array([row["wrong"] for row in rows])
    nonempty = np.array([bool(row["answer"].strip()) for row in rows])
    chosen = calibration_topics(topics)
    cal = np.isin(topics, sorted(chosen))
    test = ~cal
    test_languages = sorted(set(languages[test]))
    missing = set(test_languages) - set(languages[cal])
    if missing:
        raise ValueError(f"No calibration data for: {', '.join(sorted(missing))}")

    # Empty candidates always abstain under all three policies.
    scores = np.where(nonempty, scores, np.inf)
    pooled = conformal_threshold(scores[cal], wrong[cal] & nonempty[cal], alpha)
    by_language = per_language_thresholds(scores[cal], wrong[cal] & nonempty[cal], languages[cal], alpha)
    decisions = {
        "frozen_tydiqa": (scores <= frozen) & nonempty,
        "pooled_health": (scores <= pooled) & nonempty,
        "per_language_health": (scores <= np.array([by_language[lang] for lang in languages])) & nonempty,
    }
    groups = {"all": test, **{lang: test & (languages == lang) for lang in test_languages}}
    baseline = decisions["frozen_tydiqa"]
    rng = np.random.default_rng(seed)
    test_topics = sorted(set(topics[test]))
    # Resample entire held-out topics jointly for every language and policy.
    draws = rng.multinomial(len(test_topics), np.full(len(test_topics), 1 / len(test_topics)), size=4000)
    results = {}
    for name, answered in decisions.items():
        results[name] = {"test": summarize(answered, wrong, groups)}
        if name == "frozen_tydiqa":
            continue
        difference = answered.astype(int) - baseline
        intervals = {}
        for label, mask in groups.items():
            blocks = [(topics == topic) & mask for topic in test_topics]
            denominators = draws @ np.array([block.sum() for block in blocks])
            observed = denominators > 0
            intervals[label] = {}
            for metric, outcome in (("coverage", np.ones(len(rows), dtype=bool)),
                                    ("wrong_per_question", wrong), ("right_per_question", ~wrong)):
                counts = np.array([difference[block & outcome].sum() for block in blocks])
                rates = (draws @ counts)[observed] / denominators[observed]
                intervals[label][metric] = np.quantile(rates, [0.025, 0.975]).tolist()
        results[name]["paired_topic_bootstrap_95pct_delta"] = intervals

    def finite(value):
        return float(value) if np.isfinite(value) else None

    return {
        "joint_calibration_target": alpha,
        "topic_split": "SHA-256 of health-recalibration-v1/<topic>, first 40% for calibration",
        "calibration_topics": sorted(chosen), "test_topics": test_topics,
        "calibration_n_by_language": {lang: int((cal & (languages == lang)).sum()) for lang in test_languages},
        "thresholds": {"frozen_tydiqa": float(frozen), "pooled_health": finite(pooled),
                       "per_language_health": {lang: finite(value) for lang, value in by_language.items()}},
        "policies": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--guard", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from combine import similarities

    rows = revised_labels(read_jsonl(args.run / "answers.jsonl"))
    data = {row["id"]: row for row in read_jsonl(args.data)}
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(data):
        raise ValueError("Expected exactly one generated answer for each input health question.")
    contexts = {key: row["context"] for key, row in data.items()}
    sim = similarities(rows, contexts, args.run / "similarity.json")
    with np.load(args.run / "activations.npz") as acts:
        layer = acts["layers"].tolist().index(LAYER)
        activation = acts["answer"][:, layer]
    guard = Guard(args.guard)
    scores = guard.risk(activation, sim, np.array([row["mean_logprob"] for row in rows]))
    report = compare(rows, scores, float(guard.w["threshold_joint"]), 0.10)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
