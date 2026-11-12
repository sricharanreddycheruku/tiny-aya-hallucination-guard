"""Frozen guard: fit on probe, calibrate on calibration, report on one evaluation run."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze import column, load
from calibration import (aurc, auroc, bootstrap_ci, conformal_threshold,
                         paired_test, per_language_thresholds, selective_report, selective_threshold)
from combine import LANGS, LAYER_INDEX, load_splits, logit, similarities
from gate_probe import held_out_scores

METRICS = {"wrong_delivered": "minimize", "right_delivered": "maximize", "answered": "maximize"}


def fit_guard(train, train_acts, train_keep, contexts, train_run, layer_index):
    x = train_acts["answer"][train_keep, layer_index].astype(np.float32)
    y = column(train, "wrong")
    groups = np.array([r["language"] + "/" + r["title"] for r in train])
    out_of_fold = held_out_scores(x, y, groups)
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=500)).fit(x, y)
    similarity = similarities(train, contexts, f"{train_run}/similarity.json")
    stack = np.c_[logit(out_of_fold), similarity, column(train, "mean_logprob")]
    stacker = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(stack, y)
    return probe, stacker


def risk(probe, stacker, records, acts, index, layer_index, contexts, cache):
    x = acts["answer"][index, layer_index].astype(np.float32)
    probe_risk = probe.predict_proba(x)[:, 1]
    similarity = similarities(records, contexts, cache)
    stack = np.c_[logit(probe_risk), similarity, column(records, "mean_logprob")]
    return stacker.predict_proba(stack)[:, 1], probe_risk, similarity


def all_scores(records, run, guard, probe_risk, similarity):
    scores = {"guard": guard, "probe": probe_risk, "similarity": -similarity, "logprob": -column(records, "mean_logprob")}
    for name, fields in (("baselines", ("p_true", "agreement")), ("consensus", ("consensus",))):
        path = Path(run) / f"{name}.jsonl"
        if path.exists():
            stored = {json.loads(line)["id"]: json.loads(line) for line in open(path, encoding="utf-8")}
            for key in fields:
                scores[key] = -np.array([stored[r["id"]][key] for r in records])
    return scores


def auroc_gain(a, b, wrong):
    return auroc(a, wrong) - auroc(b, wrong)


def audit_run(name, records, answered):
    from auditkit.report import Prediction, RunResult
    from auditkit.score import Stat

    stats = {metric: Stat(metric) for metric in METRICS}
    predictions = []
    for r, spoke in zip(records, answered):
        values = {
            "wrong_delivered": float(spoke and r["wrong"]),
            "right_delivered": float(spoke and not r["wrong"]),
            "answered": float(spoke),
        }
        for metric, value in values.items():
            stats[metric].add(value)
        predictions.append(Prediction(
            run_id=name, task=r["language"], sample_id=r["id"], prompt=r["question"],
            raw_output=r["answer"] if spoke else None, parsed_answer=r["answer"] if spoke else None,
            expected=r["golds"][0], correct=not r["wrong"], score=values["wrong_delivered"],
            metadata={"scores": [
                {"name": metric, "value": value, "direction": METRICS[metric]} for metric, value in values.items()
            ]},
        ))
    headline = {metric: stat.mean for metric, stat in stats.items()}
    return RunResult(run_id=name, fingerprint=name, stats=stats, predictions=predictions, headline=headline)


def main():
    from auditkit._bootstrap import score_pairs
    from auditkit.comparison import RunComparison

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", default=["../data/questions.jsonl"])
    parser.add_argument("--train-run", default="../runs/rag_probe")
    parser.add_argument("--calibration-run", default="../runs/rag_calibration")
    parser.add_argument("--eval-run", required=True)
    parser.add_argument("--closed-run", help="closed-book answers to the same questions")
    parser.add_argument("--layer-index", type=int, default=LAYER_INDEX)
    parser.add_argument("--joint-alpha", type=float, default=0.10)
    parser.add_argument("--selective-alpha", type=float, default=0.20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    _, train, train_acts, train_keep, cal, cal_acts, keep = load_splits(args.train_run, args.calibration_run)
    contexts = {}
    for path in args.data:
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            contexts[row["id"]] = row["context"]
    probe, stacker = fit_guard(train, train_acts, train_keep, contexts, args.train_run, args.layer_index)
    cal_risk, cal_probe, cal_similarity = risk(probe, stacker, cal, cal_acts, keep, args.layer_index, contexts, f"{args.calibration_run}/similarity.json")
    cal_wrong = column(cal, "wrong")
    cal_scores = all_scores(cal, args.calibration_run, cal_risk, cal_probe, cal_similarity)
    thresholds = {
        "joint": conformal_threshold(cal_risk, cal_wrong, args.joint_alpha),
        "selective": selective_threshold(cal_risk, cal_wrong, args.selective_alpha),
    }

    records, acts = load(args.eval_run)
    everything = np.arange(len(records))
    guard, probe_risk, similarity = risk(probe, stacker, records, acts, everything, args.layer_index, contexts, f"{args.eval_run}/similarity.json")
    wrong, languages = column(records, "wrong"), column(records, "language")
    scores = all_scores(records, args.eval_run, guard, probe_risk, similarity)
    shared = [name for name in scores if name in cal_scores]
    global_rule = {
        name: {
            "joint": conformal_threshold(cal_scores[name], cal_wrong, args.joint_alpha),
            "selective": selective_threshold(cal_scores[name], cal_wrong, args.selective_alpha),
        } for name in shared
    }
    own = per_language_thresholds(cal_risk, cal_wrong, column(cal, "language"), args.joint_alpha)

    report = {"thresholds": thresholds, "n": len(records), "languages": {}}
    for language in LANGS:
        m = languages == language
        entry = {"n": int(m.sum()), "wrong_if_always_answering": float(wrong[m].mean()), "scores": {}, "guard": {}}
        for name, s in scores.items():
            low, high = bootstrap_ci(auroc, s[m], wrong[m], n_boot=1000)
            entry["scores"][name] = {"auroc": auroc(s[m], wrong[m]), "ci": [low, high], "aurc": aurc(s[m], wrong[m])}
            if name != "guard":
                entry["scores"][name]["guard_gain_ci"] = list(bootstrap_ci(auroc_gain, guard[m], s[m], wrong[m], n_boot=1000))
        for kind, t in thresholds.items():
            entry["guard"][kind] = selective_report(guard[m], wrong[m], t)
        entry["guard"]["joint_own_language_threshold"] = selective_report(guard[m], wrong[m], own[language])
        entry["global_threshold"] = {
            name: {kind: selective_report(scores[name][m], wrong[m], t) for kind, t in rules.items()}
            for name, rules in global_rule.items()
        }
        report["languages"][language] = entry

    always = audit_run("rag_always_answers", records, np.ones(len(records), dtype=bool))
    report["audit"] = {}
    for kind, t in thresholds.items():
        guarded = audit_run(f"rag_guard_{kind}", records, guard <= t)
        comparison = RunComparison(always, guarded)
        report["audit"][kind] = {metric: paired_test(score_pairs(always, guarded, metric)) for metric in METRICS}
        print(comparison.summary(), "\n", flush=True)
    if args.closed_run:
        closed, _ = load(args.closed_run)
        order = {r["id"]: r for r in closed}
        closed = [order[r["id"]] for r in records]
        before = audit_run("closed_book", closed, np.ones(len(closed), dtype=bool))
        comparison = RunComparison(before, always)
        report["audit"]["closed_to_rag"] = {metric: paired_test(score_pairs(before, always, metric)) for metric in METRICS}
        report["closed_book_wrong"] = {l: float(column(closed, "wrong")[languages == l].mean()) for l in LANGS}
        print(comparison.summary(), "\n", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    for language in LANGS:
        entry = report["languages"][language]
        print(language, entry["n"], "wrong", round(entry["wrong_if_always_answering"], 3),
              {k: round(v["auroc"], 3) for k, v in entry["scores"].items()},
              {k: {a: round(b, 3) for a, b in v.items()} for k, v in entry["guard"].items()})


if __name__ == "__main__":
    main()
