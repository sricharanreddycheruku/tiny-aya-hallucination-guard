"""Ablations on the calibration split: language transfer, features, layers, data size."""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze import column
from calibration import auroc
from combine import LANGS, LAYER_INDEX, load_splits


def fit_score(x_train, y_train, x_test):
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=500))
    return probe.fit(x_train, y_train).predict_proba(x_test)[:, 1]


def by_language(scores, wrong, languages):
    return {l: round(auroc(scores[languages == l], wrong[languages == l]), 3) for l in LANGS}


def main():
    _, train, train_acts, train_keep, cal, cal_acts, keep = load_splits()
    y_train, y_cal = column(train, "wrong"), column(cal, "wrong")
    l_train, l_cal = column(train, "language"), column(cal, "language")
    layers = train_acts["layers"].tolist()
    answer_train = train_acts["answer"][train_keep].astype(np.float32)
    answer_cal = cal_acts["answer"][keep].astype(np.float32)
    prompt_train = train_acts["prompt"][train_keep].astype(np.float32)
    prompt_cal = cal_acts["prompt"][keep].astype(np.float32)
    results = {}

    x_train, x_cal = answer_train[:, LAYER_INDEX], answer_cal[:, LAYER_INDEX]
    results["all_languages"] = by_language(fit_score(x_train, y_train, x_cal), y_cal, l_cal)
    print("all languages", results["all_languages"], flush=True)

    results["held_out_language"] = {}
    for language in LANGS:
        m = l_train != language
        scores = fit_score(x_train[m], y_train[m], x_cal[l_cal == language])
        results["held_out_language"][language] = round(auroc(scores, y_cal[l_cal == language]), 3)
    print("trained without the language", results["held_out_language"], flush=True)

    results["prompt_end_features"] = {}
    results["answer_end_features"] = {}
    for index, layer in enumerate(layers):
        results["prompt_end_features"][layer] = by_language(fit_score(prompt_train[:, index], y_train, prompt_cal[:, index]), y_cal, l_cal)
        results["answer_end_features"][layer] = by_language(fit_score(answer_train[:, index], y_train, answer_cal[:, index]), y_cal, l_cal)
        print("layer", layer, "prompt", results["prompt_end_features"][layer], "answer", results["answer_end_features"][layer], flush=True)

    results["train_fraction"] = {}
    rng = np.random.default_rng(0)
    order = rng.permutation(len(y_train))
    for fraction in (0.05, 0.1, 0.25, 0.5, 1.0):
        part = order[: int(fraction * len(order))]
        results["train_fraction"][fraction] = by_language(fit_score(x_train[part], y_train[part], x_cal), y_cal, l_cal)
        print("fraction", fraction, "rows", len(part), results["train_fraction"][fraction], flush=True)

    json.dump(results, open("../runs/ablations.json", "w"), indent=1)


if __name__ == "__main__":
    main()
