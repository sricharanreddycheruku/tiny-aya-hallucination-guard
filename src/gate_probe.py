"""Held-out probe AUROC against log-probability confidence, per language."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from calibration import auroc, bootstrap_ci


def held_out_scores(features, wrong, groups, folds=5):
    scores = np.zeros(len(wrong))
    for train, test in GroupKFold(n_splits=folds).split(features, wrong, groups):
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=500))
        probe.fit(features[train], wrong[train])
        scores[test] = probe.predict_proba(features[test])[:, 1]
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/gate_fire")
    args = parser.parse_args()
    run = Path(args.run)

    records = [json.loads(line) for line in open(run / "answers.jsonl", encoding="utf-8")]
    acts = np.load(run / "activations.npz")
    languages = np.array([r["language"] for r in records])
    wrong = np.array([r["wrong"] for r in records])
    groups = np.array([r["language"] + "/" + r["title"] for r in records])
    logprob_risk = -np.array([r["mean_logprob"] for r in records])

    print("language  n  wrong_rate  exact  contains  empty  logprob_auroc")
    for language in np.unique(languages):
        m = languages == language
        print(
            language, int(m.sum()), round(wrong[m].mean(), 3),
            round(np.mean([r["exact"] for r, k in zip(records, m) if k]), 3),
            round(np.mean([r["contains"] for r, k in zip(records, m) if k]), 3),
            int(sum(r["n_tokens"] == 0 for r, k in zip(records, m) if k)),
            round(auroc(logprob_risk[m], wrong[m]), 3),
        )

    results = {}
    for position in ("prompt", "answer"):
        for index, layer in enumerate(acts["layers"].tolist()):
            features = acts[position][:, index].astype(np.float32)
            scores = held_out_scores(features, wrong, groups)
            row = {l: auroc(scores[languages == l], wrong[languages == l]) for l in np.unique(languages).tolist()}
            results[f"{position}_{layer}"] = row
            np.save(run / f"scores_{position}_{layer}.npy", scores)
            print(position, layer, {k: round(v, 3) for k, v in row.items()}, flush=True)

    best = max(results, key=lambda k: min(results[k].get(l, 0) for l in ("en", "te")))
    scores = np.load(run / f"scores_{best}.npy")
    print("selected on en/te minimum:", best)
    for language in np.unique(languages):
        m = languages == language
        print(
            language,
            "probe", round(auroc(scores[m], wrong[m]), 3),
            [round(x, 3) for x in bootstrap_ci(auroc, scores[m], wrong[m], n_boot=500)],
            "logprob", round(auroc(logprob_risk[m], wrong[m]), 3),
            [round(x, 3) for x in bootstrap_ci(auroc, logprob_risk[m], wrong[m], n_boot=500)],
        )
    (run / "probe_results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
