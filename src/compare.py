"""Compare the probe with every baseline on the calibration split."""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze import column, keep_rows, load
from calibration import aurc, auroc, bootstrap_ci

LAYER_INDEX = 5  # layer 24, chosen on the probe split

rows = [json.loads(line) for line in open("../data/questions.jsonl", encoding="utf-8")]
test_questions = {r["question"] for r in rows if r["split"] == "test"}
meta = {r["id"]: r for r in rows}
cal, cal_acts = load("../runs/rag_calibration")
keep = keep_rows(cal, test_questions)
cal = [cal[i] for i in keep]
probe_rows, probe_acts = load("../runs/rag_probe")
probe_keep = keep_rows(probe_rows, test_questions | {r["question"] for r in cal})
probe_rows = [probe_rows[i] for i in probe_keep]

model = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=500))
model.fit(probe_acts["answer"][probe_keep, LAYER_INDEX].astype(np.float32), column(probe_rows, "wrong"))
risk = model.predict_proba(cal_acts["answer"][keep, LAYER_INDEX].astype(np.float32))[:, 1]

extra = {}
for line in open("../runs/rag_calibration/baselines.jsonl"):
    r = json.loads(line)
    extra[r["id"]] = r
scores = {
    "probe": risk,
    "logprob": -column(cal, "mean_logprob"),
    "similarity": -np.array([extra[r["id"]]["similarity"] for r in cal]),
    "p_true": -np.array([extra[r["id"]]["p_true"] for r in cal]),
    "agreement": -np.array([extra[r["id"]]["agreement"] for r in cal]),
}
wrong = column(cal, "wrong")
languages = column(cal, "language")
in_context = np.array([meta[r["id"]]["answer_in_context"] for r in cal])

print("AUROC [95% CI] and AURC by language")
for language in ("en", "te", "bn", "sw"):
    m = languages == language
    for name, s in scores.items():
        low, high = bootstrap_ci(auroc, s[m], wrong[m], n_boot=500)
        print(language, name, round(auroc(s[m], wrong[m]), 3), [round(low, 3), round(high, 3)], "aurc", round(aurc(s[m], wrong[m]), 3))

print("\nAUROC inside each stratum: answer present in passage / absent")
for language in ("en", "te", "bn", "sw"):
    m = languages == language
    line = [language, "n_present", int((m & in_context).sum()), "n_absent", int((m & ~in_context).sum()), "right_absent", int((m & ~in_context & ~wrong).sum())]
    for name, s in scores.items():
        line += [name, round(auroc(s[m & in_context], wrong[m & in_context]), 3), round(auroc(s[m & ~in_context], wrong[m & ~in_context]), 3)]
    print(*line)
