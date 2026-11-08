"""Check the demo pipeline reproduces stored experiment values on real rows."""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "../space")
from pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="../runs/rag_calibration")
    parser.add_argument("--folder", default="../space_build")
    parser.add_argument("--per-language", type=int, default=15)
    args = parser.parse_args()

    contexts = {}
    for line in open("../data/questions.jsonl", encoding="utf-8"):
        row = json.loads(line)
        contexts[row["id"]] = row["context"]
    records = [json.loads(line) for line in open(f"{args.run}/answers.jsonl", encoding="utf-8")]
    stored_similarity = json.load(open(f"{args.run}/similarity.json"))
    pipeline = Pipeline(args.folder)
    counts, same_answer, logprob_gap, risk_gap = {}, [], [], []
    acts = np.load(f"{args.run}/activations.npz")["answer"]
    for i, r in enumerate(records):
        if counts.get(r["language"], 0) >= args.per_language or r["id"] not in stored_similarity or len(contexts[r["id"]]) > 3000:
            continue
        counts[r["language"]] = counts.get(r["language"], 0) + 1
        got = pipeline.respond(r["question"], r["language"], {"text": contexts[r["id"]]})
        stored = pipeline.guard.risk(acts[i : i + 1, 5].astype(np.float32), np.array([stored_similarity[r["id"]]]), np.array([r["mean_logprob"]]))[0]
        same_answer.append(got["answer"] == r["answer"])
        logprob_gap.append(abs(got["mean_logprob"] - r["mean_logprob"]))
        risk_gap.append(abs(got["risk"] - stored))
    print("rows", len(same_answer), "same answer", np.mean(same_answer), "logprob gap median", np.median(logprob_gap),
          "risk gap median", np.median(risk_gap), "max", np.max(risk_gap))


if __name__ == "__main__":
    main()
