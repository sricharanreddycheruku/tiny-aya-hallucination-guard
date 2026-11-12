"""Build probe, calibration and test questions with a retrieved passage each.

Passages come from all TyDiQA contexts of the same language. For a share of
questions the source article is removed from the search, so the retriever
returns related but possibly unhelpful text, as a real knowledge base would.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from scoring import normalize

ENCODER = "intfloat/multilingual-e5-small"


@torch.no_grad()
def embed(texts, tokenizer, encoder, prefix, batch_size=256):
    chunks = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(
            [prefix + t for t in texts[start : start + batch_size]],
            padding=True, truncation=True, max_length=256, return_tensors="pt",
        ).to(encoder.device)
        hidden = encoder(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        chunks.append(F.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1))
    return torch.cat(chunks)


def article_split(title):
    return "probe" if int(hashlib.sha1(title.encode()).hexdigest(), 16) % 10 < 7 else "calibration"


def main():
    from datasets import load_dataset

    from gate import LANGUAGES

    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-size", type=int, default=1500)
    parser.add_argument("--calibration-size", type=int, default=700)
    parser.add_argument("--uncovered", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="../data/questions.jsonl")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(ENCODER)
    encoder = AutoModel.from_pretrained(ENCODER).to("cuda").eval()
    train = load_dataset("google-research-datasets/tydiqa", "secondary_task", split="train")
    test = load_dataset("google-research-datasets/tydiqa", "secondary_task", split="validation")
    rng = random.Random(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as out:
        for name, language in LANGUAGES.items():
            train_rows = [r for r in train if r["id"].startswith(name + "-")]
            test_rows = [r for r in test if r["id"].startswith(name + "-")]
            test_titles = {r["title"] for r in test_rows}

            passages = sorted({(r["title"], r["context"]) for r in train_rows + test_rows})
            titles = [p[0] for p in passages]
            passage_vectors = embed([p[1] for p in passages], tokenizer, encoder, "passage: ")

            usable = [r for r in train_rows if r["title"] not in test_titles]
            rng.shuffle(usable)
            chosen = [(r, "test") for r in test_rows]
            for split, size in (("probe", args.probe_size), ("calibration", args.calibration_size)):
                chosen += [(r, split) for r in usable if article_split(r["title"]) == split][:size]

            query_vectors = embed([r["question"] for r, _ in chosen], tokenizer, encoder, "query: ")
            hits = {"covered": [], "uncovered": []}
            for (row, split), vector in zip(chosen, query_vectors):
                covered = rng.random() >= args.uncovered
                scores = passage_vectors @ vector
                if not covered:
                    blocked = [i for i, t in enumerate(titles) if t == row["title"]]
                    scores[blocked] = -1.0
                best = int(scores.argmax())
                context = passages[best][1]
                golds = sorted(set(row["answers"]["text"]))
                hit = any(normalize(g, language) in normalize(context, language) for g in golds)
                hits["covered" if covered else "uncovered"].append(hit)
                out.write(json.dumps({
                    "id": row["id"], "language": language, "title": row["title"],
                    "question": row["question"], "golds": golds, "context": context,
                    "split": split, "covered": covered, "answer_in_context": hit,
                    "same_article": passages[best][0] == row["title"],
                }, ensure_ascii=False) + "\n")
            print(language, len(chosen), "passages", len(passages), {
                k: round(sum(v) / max(len(v), 1), 3) for k, v in hits.items()
            }, flush=True)


if __name__ == "__main__":
    main()
