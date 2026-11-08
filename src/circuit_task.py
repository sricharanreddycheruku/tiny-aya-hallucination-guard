"""Counterfactual evidence task for CircuitKIT, built from the probe split only.

Each pair holds one question and one passage. The corrupt prompt swaps the gold
answer in the passage for another answer of the same token length, so the logit
difference measures whether the model's answer follows the evidence.
"""
import argparse
import csv
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

from gate import INSTRUCTION

WINDOW_CHARS = 200
MAX_TOKENS = 220
USER_TURN = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>"


def window(context, gold):
    start = context.find(gold)
    left = max(0, start - WINDOW_CHARS // 2)
    return context[left:start + len(gold) + WINDOW_CHARS // 2]


def render(tokenizer, passage, question):
    content = f"{INSTRUCTION}Passage: {passage}\n\nQuestion: {question}"
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
    return full[full.index(USER_TURN):]


def build_pairs(rows, tokenizer, per_language, seed):
    rng = random.Random(seed)
    pairs = []
    for language in sorted({row["language"] for row in rows}):
        usable = []
        for row in rows:
            if row["language"] != language:
                continue
            golds = [gold for gold in row["golds"] if gold and gold in row["context"]]
            if golds:
                usable.append((row, golds[0]))
        rng.shuffle(usable)
        by_length = {}
        for _, gold in usable:
            ids = tokenizer.encode(gold, add_special_tokens=False)
            by_length.setdefault(len(ids), []).append((gold, ids[0]))
        kept = 0
        for row, gold in usable:
            if kept == per_language:
                break
            ids = tokenizer.encode(gold, add_special_tokens=False)
            peers = [g for g, first in by_length[len(ids)] if first != ids[0] and g != gold]
            if not peers:
                continue
            other = rng.choice(peers)
            passage = window(row["context"], gold)
            clean = render(tokenizer, passage, row["question"])
            corrupt = render(tokenizer, passage.replace(gold, other), row["question"])
            length = len(tokenizer.encode(clean))
            if length != len(tokenizer.encode(corrupt)) or length > MAX_TOKENS:
                continue
            pairs.append({
                "clean_prompt": clean,
                "corrupt_prompt": corrupt,
                "clean_answer": gold,
                "corrupt_answer": other,
                "language": language,
            })
            kept += 1
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="../data/questions.jsonl")
    parser.add_argument("--model", default="CohereLabs/tiny-aya-fire")
    parser.add_argument("--per-language", type=int, default=64)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(line) for line in open(args.questions, encoding="utf-8")]
    rows = [row for row in rows if row["split"] == "probe" and row["answer_in_context"]]
    pairs = build_pairs(rows, tokenizer, args.per_language, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    counts = {}
    for pair in pairs:
        counts[pair["language"]] = counts.get(pair["language"], 0) + 1
    print(counts)


if __name__ == "__main__":
    main()
