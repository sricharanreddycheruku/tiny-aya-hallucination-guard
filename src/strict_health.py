"""Stricter health subset: rescore gold answers with the judge, keep those at or above the bar.

The planted-answer check in gate_check.py showed that 0.85 rejects almost all wrong
answers. Selection uses only the gold answer and its source text, never model output.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from curatorkit.gates.hallucination import HallucinationGate

from curate import LocalLLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/health_big.jsonl")
    parser.add_argument("--run", default="../runs/rag_health_big")
    parser.add_argument("--out", default="../runs/rag_health_strict")
    parser.add_argument("--judge", default="CohereLabs/tiny-aya-global")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    saved = out / "grounding.json"
    if saved.exists():
        scores = json.load(open(saved))
    else:
        judge = LocalLLM(args.judge, max_tokens=96)
        gate = HallucinationGate(llm=judge)
        scores = {}
        for i, row in enumerate(rows):
            scores[row["id"]] = gate._evaluate_grounding(row["source_text"], row["question"], row["golds"][0])[0]
            if i % 100 == 0:
                print(i, len(rows), flush=True)
        judge.close()
        json.dump(scores, open(saved, "w"), indent=0)

    lines = open(f"{args.run}/answers.jsonl", encoding="utf-8").readlines()
    keep = [i for i, line in enumerate(lines) if scores[json.loads(line)["id"]] >= args.threshold]
    with open(out / "answers.jsonl", "w", encoding="utf-8") as handle:
        handle.writelines(lines[i] for i in keep)
    acts = np.load(f"{args.run}/activations.npz")
    arrays = {key: acts[key][keep] if len(acts[key]) == len(lines) else acts[key] for key in acts.files}
    np.savez(out / "activations.npz", **arrays)
    for name in ("baselines.jsonl", "consensus.jsonl"):
        shutil.copy(f"{args.run}/{name}", out / name)
    print("kept", len(keep), "of", len(lines), flush=True)


if __name__ == "__main__":
    main()
