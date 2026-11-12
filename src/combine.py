"""Stack the probe with free signals, fitted on the probe split only."""
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze import column, keep_rows, load
from calibration import (aurc, auroc, bootstrap_ci, conformal_threshold,
                         selective_report, selective_threshold)
from gate_probe import held_out_scores

LAYER_INDEX = 5  # layer 24, chosen on the probe split
LANGS = ("en", "te", "bn", "sw")


def similarities(records, contexts, cache):
    cache = Path(cache)
    if cache.exists():
        stored = json.loads(cache.read_text(encoding="utf-8"))
        if all(r["id"] in stored for r in records):
            return np.array([stored[r["id"]] for r in records])
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    from build_rag import ENCODER, embed

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 256 if device == "cuda" else 32
    tokenizer = AutoTokenizer.from_pretrained(ENCODER)
    encoder = AutoModel.from_pretrained(ENCODER).to(device).eval()
    questions = embed([r["question"] for r in records], tokenizer, encoder, "query: ", batch_size)
    passages = embed([contexts[r["id"]] for r in records], tokenizer, encoder, "passage: ", batch_size)
    values = F.cosine_similarity(questions, passages).tolist()
    del encoder
    if device == "cuda":
        torch.cuda.empty_cache()
    cache.write_text(json.dumps({r["id"]: v for r, v in zip(records, values)}), encoding="utf-8")
    return np.array(values)


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def load_splits(train_run="../runs/rag_probe", calibration_run="../runs/rag_calibration",
                data="../data/questions.jsonl"):
    with open(data, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    test_questions = {r["question"] for r in rows if r["split"] == "test"}
    cal, cal_acts = load(calibration_run)
    keep = keep_rows(cal, test_questions)
    cal = [cal[i] for i in keep]
    train, train_acts = load(train_run)
    train_keep = keep_rows(train, test_questions | {r["question"] for r in cal})
    train = [train[i] for i in train_keep]
    return rows, train, train_acts, train_keep, cal, cal_acts, keep


def main():
    rows, train, train_acts, train_keep, cal, cal_acts, keep = load_splits()
    meta = {r["id"]: r for r in rows}
    contexts = {r["id"]: r["context"] for r in rows}

    x_train = train_acts["answer"][train_keep, LAYER_INDEX].astype(np.float32)
    x_cal = cal_acts["answer"][keep, LAYER_INDEX].astype(np.float32)
    y_train, y_cal = column(train, "wrong"), column(cal, "wrong")
    groups = np.array([r["language"] + "/" + r["title"] for r in train])
    out_of_fold = held_out_scores(x_train, y_train, groups)
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=500)).fit(x_train, y_train)
    probe_cal = probe.predict_proba(x_cal)[:, 1]

    sim_train = similarities(train, contexts, "../runs/rag_probe/similarity.json")
    sim_cal = similarities(cal, contexts, "../runs/rag_calibration/similarity.json")
    stack_train = np.c_[logit(out_of_fold), sim_train, column(train, "mean_logprob")]
    stack_cal = np.c_[logit(probe_cal), sim_cal, column(cal, "mean_logprob")]
    stacker = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(stack_train, y_train)
    print("stacker weights [probe, similarity, logprob]", stacker[-1].coef_.round(3).tolist())
    combined = stacker.predict_proba(stack_cal)[:, 1]

    scores = {
        "combined": combined, "probe": probe_cal, "similarity": -sim_cal,
        "logprob": -column(cal, "mean_logprob"),
    }
    languages = column(cal, "language")
    hit = np.array([meta[r["id"]]["answer_in_context"] for r in cal])

    print("\nAUROC [95% CI] / AURC on calibration")
    for language in LANGS:
        m = languages == language
        for name, s in scores.items():
            low, high = bootstrap_ci(auroc, s[m], y_cal[m], n_boot=500)
            print(language, name, round(auroc(s[m], y_cal[m]), 3), [round(low, 3), round(high, 3)], "aurc", round(aurc(s[m], y_cal[m]), 3))

    print("\nwhy similarity works: AUROC of similarity for 'answer not in passage', hit rate, mean sim present/absent")
    for language in LANGS:
        m = languages == language
        print(
            language, round(auroc(-sim_cal[m], ~hit[m]), 3), "hit", round(hit[m].mean(), 3),
            "sim", round(sim_cal[m & hit].mean(), 3), round(sim_cal[m & ~hit].mean(), 3),
            "wrong|present", round(y_cal[m & hit].mean(), 3), "wrong|absent", round(y_cal[m & ~hit].mean(), 3),
        )

    half = np.array([int(hashlib.sha1(r["title"].encode()).hexdigest(), 16) % 2 for r in cal]) == 0
    for kind, rule, alphas in (
        ("joint", conformal_threshold, (0.05, 0.10)),
        ("selective", selective_threshold, (0.10, 0.20)),
    ):
        for alpha in alphas:
            print(f"\ntarget {kind} risk {alpha}, one global threshold from half the articles")
            for name, s in scores.items():
                t = rule(s[half], y_cal[half], alpha)
                line = [name]
                for language in LANGS:
                    m = (languages == language) & ~half
                    r = selective_report(s[m], y_cal[m], t)
                    line += [language, "cov", round(r["coverage"], 3), "joint", round(r["joint_risk"], 3), "sel", round(r["selective_risk"], 3)]
                print(*line)


if __name__ == "__main__":
    main()
