"""Merge sampled health rounds into one set, without repeats or rows from the existing set."""
import argparse
import json
import random
from collections import defaultdict

from scoring import normalize


def merge(paths, exclude, per_language, seed=0):
    seen = {(r["language"], normalize(r["question"], r["language"])) for r in exclude}
    grouped = defaultdict(list)
    for path in paths:
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            key = (row["language"], normalize(row["question"], row["language"]))
            if key not in seen:
                seen.add(key)
                grouped[row["language"]].append(row)
    rng = random.Random(seed)
    merged = []
    for language in sorted(grouped):
        rows = grouped[language]
        rng.shuffle(rows)
        for i, row in enumerate(rows[:per_language]):
            merged.append(dict(row, id=f"health2-{language}-{i}"))
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", nargs="+", required=True)
    parser.add_argument("--exclude", default="../data/health_big.jsonl")
    parser.add_argument("--per-language", type=int, default=1500)
    parser.add_argument("--out", default="../data/health_more.jsonl")
    args = parser.parse_args()

    exclude = [json.loads(line) for line in open(args.exclude, encoding="utf-8")]
    rows = merge(args.rounds, exclude, args.per_language)
    with open(args.out, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = defaultdict(int)
    for row in rows:
        counts[row["language"]] += 1
    print(dict(counts), flush=True)


if __name__ == "__main__":
    main()
