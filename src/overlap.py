"""How much of the evidence circuit do the four languages share?

Compares per-language EAP-IG attention head scores: rank correlation, overlap of the
top heads against a shuffled baseline, and the block carrying the most importance.
"""
import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

LANGS = ("en", "te", "bn", "sw")


def heads(path):
    stored = json.load(open(path))["node_scores"]
    return {name: abs(value) for name, value in stored.items() if name.startswith("A")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="../runs/circuit")
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--shuffles", type=int, default=2000)
    parser.add_argument("--out", default="../runs/circuit/overlap.json")
    args = parser.parse_args()

    scores = {language: heads(f"{args.dir}/evidence_{language}_scores.json") for language in LANGS}
    names = sorted(set.intersection(*(set(s) for s in scores.values())))
    matrix = {language: np.array([scores[language][n] for n in names]) for language in LANGS}
    top = {language: set(np.argsort(-matrix[language])[: args.top].tolist()) for language in LANGS}

    rng = np.random.default_rng(0)
    chance = np.mean([
        len(set(rng.choice(len(names), args.top, replace=False).tolist()) & set(rng.choice(len(names), args.top, replace=False).tolist()))
        for _ in range(args.shuffles)
    ])
    report = {"heads": len(names), "top": args.top, "chance_shared": float(chance), "pairs": {}, "peak_block": {}, "shared_by_all": len(set.intersection(*top.values()))}
    for a, b in combinations(LANGS, 2):
        report["pairs"][f"{a}-{b}"] = {
            "spearman": float(spearmanr(matrix[a], matrix[b]).statistic),
            "shared_top": len(top[a] & top[b]),
        }
    for language in LANGS:
        mass = defaultdict(float)
        for name, value in zip(names, matrix[language]):
            mass[int(name[1:].split(".")[0])] += value
        ranked = sorted(mass, key=mass.get, reverse=True)
        report["peak_block"][language] = ranked[:3]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
