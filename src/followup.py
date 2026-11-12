"""Evaluate majority coverage and independent-pilot calibration on saved RAG runs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from calibration import certify_selective_threshold, coverage_threshold
from scoring import is_wrong

TARGETS = (0.55, 0.60)
ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def revised_labels(records):
    return [
        dict(row, wrong=is_wrong(row["answer"], row["golds"], row["language"]))
        for row in records
    ]


def metrics(answered, wrong):
    answered = np.asarray(answered, dtype=bool)
    wrong = np.asarray(wrong, dtype=bool)
    n = len(answered)
    delivered = int(answered.sum())
    errors = int((answered & wrong).sum())
    return {
        "n": n, "answered": delivered, "wrong_delivered": errors, "right_delivered": delivered - errors,
        "coverage": delivered / n if n else None,
        "wrong_per_question": errors / n if n else None,
        "right_per_question": (delivered - errors) / n if n else None,
        "wrong_among_answered": errors / delivered if delivered else None,
    }


def slices(records):
    languages = np.array([row["language"] for row in records])
    groups = {"all": np.ones(len(records), dtype=bool)}
    groups.update({language: languages == language for language in sorted(set(languages))})
    if all("covered" in row and "answer_in_context" in row for row in records):
        source_available = np.array([row["covered"] for row in records], dtype=bool)
        gold_in_passage = np.array([row["answer_in_context"] for row in records], dtype=bool)
        groups.update(
            source_article_available=source_available,
            source_article_excluded=~source_available,
            gold_string_present=gold_in_passage,
            gold_string_absent=~gold_in_passage,
        )
    return groups


def summarize(answered, wrong, groups):
    return {name: metrics(answered[mask], wrong[mask]) for name, mask in groups.items()}


def pilot_split(records):
    """Assign whole articles to a pilot or certification partition."""
    return np.array([
        hashlib.sha256(f'{row["language"]}/{row["title"]}'.encode()).digest()[0] < 128
        for row in records
    ])


def evaluate(calibration, cal_scores, test, test_scores):
    cal_wrong = np.array([row["wrong"] for row in calibration])
    wrong = np.array([row["wrong"] for row in test])
    nonempty = np.array([bool(row["answer"].strip()) for row in test])
    groups = slices(test)
    report = {
        "always_answer": summarize(np.ones(len(test), dtype=bool), wrong, groups),
        "majority": {},
    }
    decisions = {}
    for target in TARGETS:
        point = {}
        for name, scores in test_scores.items():
            threshold = coverage_threshold(cal_scores[name], target)
            answered = (scores <= threshold) & nonempty
            point[name] = {
                "threshold": threshold,
                "calibration_coverage": float(np.mean(cal_scores[name] <= threshold)),
                "test": summarize(answered, wrong, groups),
            }
            decisions[target, name] = answered
        report["majority"][str(target)] = point

    pilot = pilot_split(calibration)
    if not pilot.any() or pilot.all():
        raise ValueError("Calibration data must contain articles in both pilot partitions.")
    guard_cal = cal_scores["guard"]
    certified = certify_selective_threshold(guard_cal[pilot], guard_cal[~pilot], cal_wrong[~pilot])
    report["independent_pilot"] = {
        "pilot_n": int(pilot.sum()), "certification_n": int((~pilot).sum()),
        "threshold": certified if np.isfinite(certified) else None,
        "test": summarize((test_scores["guard"] <= certified) & nonempty, wrong, groups),
    }
    return report, decisions


def plot_risk_coverage(path, records, scores_by_method):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nonempty = np.array([bool(row["answer"].strip()) for row in records])
    wrong = np.array([row["wrong"] for row in records])[nonempty]
    if not len(wrong):
        raise ValueError("A risk-coverage plot requires at least one nonempty candidate.")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, scores in scores_by_method.items():
        scores = scores[nonempty]
        order = np.argsort(scores, kind="stable")
        sorted_scores = scores[order]
        end = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
        count = end + 1
        ax.plot(count / len(records), np.cumsum(wrong[order])[end] / count, label=name.replace("_", " "))
    ax.set(xlabel="Fraction of all questions answered", ylabel="Wrong among delivered answers")
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_review(path, records, decisions, contexts, per_group=12):
    """Write a reproducible sample for factual and passage-support annotation."""
    fields = ["id", "language", "question", "passage", "candidate", "reference",
              "factually_correct", "supported_by_passage", "review_notes"]
    sample = []
    for language in sorted({row["language"] for row in records}):
        for delivered in (False, True):
            group = [row for row, spoke in zip(records, decisions)
                     if row["language"] == language and spoke == delivered]
            group.sort(key=lambda row: hashlib.sha256(row["id"].encode()).digest())
            sample.extend(group[:per_group])
    sample.sort(key=lambda row: hashlib.sha256(f'review/{row["id"]}'.encode()).digest())
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sample:
            writer.writerow({
                "id": row["id"], "language": row["language"], "question": row["question"],
                "passage": contexts[row["id"]], "candidate": row["answer"],
                "reference": " | ".join(row["golds"]),
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "questions.jsonl")
    parser.add_argument("--train-run", type=Path, default=ROOT / "runs" / "rag_probe")
    parser.add_argument("--calibration-run", type=Path, default=ROOT / "runs" / "rag_calibration")
    parser.add_argument("--test-run", type=Path, default=ROOT / "runs" / "rag_test")
    parser.add_argument("--frozen-guard", type=Path, default=ROOT / "space_build" / "guard_fire.npz")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "followup")
    args = parser.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from analyze import column, load
    from combine import LAYER_INDEX, load_splits, similarities
    from final import fit_guard, risk
    from guard import LAYER, Guard

    rows, train, train_acts, train_keep, cal, cal_acts, cal_keep = load_splits(
        args.train_run, args.calibration_run, args.data
    )
    contexts = {row["id"]: row["context"] for row in rows}
    train, cal = revised_labels(train), revised_labels(cal)
    original_test, test_acts = load(args.test_run)
    test = revised_labels(original_test)
    for name, acts in (("probe", train_acts), ("calibration", cal_acts), ("test", test_acts)):
        if len(acts["layers"]) <= LAYER_INDEX or acts["layers"][LAYER_INDEX] != LAYER:
            raise ValueError(f"{name} needs the default activation layout with layer {LAYER} at index {LAYER_INDEX}.")

    probe, stacker = fit_guard(train, train_acts, train_keep, contexts, args.train_run, LAYER_INDEX)
    train_sim = similarities(train, contexts, f"{args.train_run}/similarity.json")
    no_activation = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    no_activation.fit(np.c_[train_sim, column(train, "mean_logprob")], column(train, "wrong"))

    def score(records, acts, index, run):
        guard, only_probe, sim = risk(
            probe, stacker, records, acts, index, LAYER_INDEX, contexts, f"{run}/similarity.json"
        )
        return {
            "guard": guard, "probe_only": only_probe,
            "no_activation": no_activation.predict_proba(np.c_[sim, column(records, "mean_logprob")])[:, 1],
        }

    cal_scores = score(cal, cal_acts, cal_keep, args.calibration_run)
    test_scores = score(test, test_acts, np.arange(len(test)), args.test_run)
    report, decisions = evaluate(cal, cal_scores, test, test_scores)
    frozen_guard = Guard(args.frozen_guard)
    test_sim = similarities(test, contexts, f"{args.test_run}/similarity.json")
    frozen_scores = frozen_guard.risk(test_acts["answer"][:, LAYER_INDEX], test_sim, column(test, "mean_logprob"))
    frozen_delivered = frozen_guard.decide(frozen_scores) & np.array([bool(row["answer"].strip()) for row in test])
    report["frozen_joint_replay"] = {
        "stored_labels": summarize(frozen_delivered, column(original_test, "wrong"), slices(test)),
        "revised_labels": summarize(frozen_delivered, column(test, "wrong"), slices(test)),
    }
    report["labels"] = "revised automatic scorer"
    report["analysis"] = "follow-up to the original locked benchmark"
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot_risk_coverage(out / "risk_coverage.png", test, test_scores)
    write_review(out / "review.csv", test, decisions[0.60, "guard"], contexts)


if __name__ == "__main__":
    main()
