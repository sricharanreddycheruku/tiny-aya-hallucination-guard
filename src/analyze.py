"""Train the risk probe and compare global with per-language thresholds.

Layer choice uses the probe split only. Thresholds come from one half of the
calibration articles and are checked on the other half. The test split is read
only for its question texts, which are removed from the other splits.
"""
import argparse
import hashlib
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from calibration import (aurc, auroc, bootstrap_ci, conformal_threshold,
                         per_language_thresholds, selective_report)
from gate_probe import held_out_scores


def load(run):
    records = [json.loads(line) for line in open(f"{run}/answers.jsonl", encoding="utf-8")]
    return records, np.load(f"{run}/activations.npz")


def column(records, key):
    return np.array([r[key] for r in records])


def keep_rows(records, banned):
    seen, keep = set(), []
    for i, r in enumerate(records):
        if r["question"] not in banned and r["question"] not in seen:
            seen.add(r["question"])
            keep.append(i)
    return np.array(keep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/questions.jsonl")
    parser.add_argument("--probe-run", default="../runs/rag_probe")
    parser.add_argument("--calibration-run", default="../runs/rag_calibration")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.10])
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    test_questions = {r["question"] for r in rows if r["split"] == "test"}
    cal_records, cal_acts = load(args.calibration_run)
    cal_keep = keep_rows(cal_records, test_questions)
    cal_records = [cal_records[i] for i in cal_keep]
    probe_records, probe_acts = load(args.probe_run)
    probe_keep = keep_rows(probe_records, test_questions | {r["question"] for r in cal_records})
    probe_records = [probe_records[i] for i in probe_keep]

    wrong = column(probe_records, "wrong")
    languages = column(probe_records, "language")
    groups = np.array([r["language"] + "/" + r["title"] for r in probe_records])
    print("rows probe", len(probe_records), "calibration", len(cal_records))
    print("language wrong_rate wrong|answer_in_context wrong|not_in_context")
    hit = column(probe_records, "answer_in_context")
    for language in np.unique(languages):
        m = languages == language
        print(language, round(wrong[m].mean(), 3), round(wrong[m & hit].mean(), 3), round(wrong[m & ~hit].mean(), 3))

    layers = probe_acts["layers"].tolist()
    best_index, best_value = 0, -1.0
    for index, layer in enumerate(layers):
        features = probe_acts["answer"][probe_keep, index].astype(np.float32)
        scores = held_out_scores(features, wrong, groups)
        per_language = {l: auroc(scores[languages == l], wrong[languages == l]) for l in np.unique(languages).tolist()}
        print("layer", layer, {k: round(v, 3) for k, v in per_language.items()}, flush=True)
        value = min(per_language["en"], per_language["te"])
        if value > best_value:
            best_index, best_value = index, value
    print("chosen layer", layers[best_index])

    probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=500))
    probe.fit(probe_acts["answer"][probe_keep, best_index].astype(np.float32), wrong)
    risk = probe.predict_proba(cal_acts["answer"][cal_keep, best_index].astype(np.float32))[:, 1]
    logprob_risk = -column(cal_records, "mean_logprob")
    cal_wrong = column(cal_records, "wrong")
    cal_languages = column(cal_records, "language")
    half = np.array([int(hashlib.sha1(r["title"].encode()).hexdigest(), 16) % 2 for r in cal_records]) == 0

    print("\ncalibration split, probe against log-probability")
    for language in np.unique(cal_languages):
        m = cal_languages == language
        print(
            language, "wrong", round(cal_wrong[m].mean(), 3),
            "probe auroc", round(auroc(risk[m], cal_wrong[m]), 3),
            [round(x, 3) for x in bootstrap_ci(auroc, risk[m], cal_wrong[m], n_boot=500)],
            "aurc", round(aurc(risk[m], cal_wrong[m]), 3),
            "| logprob auroc", round(auroc(logprob_risk[m], cal_wrong[m]), 3),
            [round(x, 3) for x in bootstrap_ci(auroc, logprob_risk[m], cal_wrong[m], n_boot=500)],
            "aurc", round(aurc(logprob_risk[m], cal_wrong[m]), 3),
        )

    for alpha in args.alphas:
        print(f"\ntarget joint risk {alpha}: thresholds from one half, measured on the other")
        for name, score in (("probe", risk), ("logprob", logprob_risk)):
            pooled = conformal_threshold(score[half], cal_wrong[half], alpha)
            own = per_language_thresholds(score[half], cal_wrong[half], cal_languages[half], alpha)
            for language in np.unique(cal_languages):
                m = (cal_languages == language) & ~half
                g = selective_report(score[m], cal_wrong[m], pooled)
                p = selective_report(score[m], cal_wrong[m], own[language])
                print(
                    name, language, "n", int(m.sum()),
                    "global risk", round(g["joint_risk"], 3), "coverage", round(g["coverage"], 3),
                    "| per-language risk", round(p["joint_risk"], 3), "coverage", round(p["coverage"], 3),
                    "selective", round(p["selective_risk"], 3),
                )


if __name__ == "__main__":
    main()
