"""Seconds per query on one GPU, one question at a time, for the guard and the competing scores."""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from baselines import agreement, p_true
from build_rag import ENCODER, embed
from consensus import answers as pivot_answers
from gate import run_batch


def timed(function):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return time.perf_counter() - started, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="CohereLabs/tiny-aya-fire")
    parser.add_argument("--data", default="../data/questions.jsonl")
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--per-language", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--out", default="../runs/latency.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    rows = [r for r in rows if r["split"] == args.split]
    picked = []
    for language in sorted({r["language"] for r in rows}):
        picked += rng.sample([r for r in rows if r["language"] == language], args.per_language)
    rng.shuffle(picked)

    token = os.environ["HF_TOKEN"]
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, token=token, dtype=torch.bfloat16, device_map="cuda").eval()
    encoder_tokenizer = AutoTokenizer.from_pretrained(ENCODER)
    encoder = AutoModel.from_pretrained(ENCODER).to("cuda").eval()
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    stop_ids = torch.tensor(sorted(set(eos + [tokenizer.pad_token_id])), device=model.device)
    yes = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no = tokenizer.encode("No", add_special_tokens=False)[0]
    weights = torch.randn(2048, device="cuda")

    @torch.no_grad()
    def plain(row):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Answer the question in the same language, using only a few words.\n\nPassage: {row['context']}\n\nQuestion: {row['question']}"}],
            tokenize=False, add_generation_prompt=True,
        )
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        return model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)

    @torch.no_grad()
    def guard(row):
        answer, _, _, _, _, acts = run_batch(model, tokenizer, [row], args.max_new_tokens, stop_ids, True, [24])
        question = embed([row["question"]], encoder_tokenizer, encoder, "query: ")
        passage = embed([row["context"]], encoder_tokenizer, encoder, "passage: ")
        F.cosine_similarity(question, passage)
        torch.tensor(acts[0, 0], device="cuda", dtype=torch.float32) @ weights
        return answer[0]

    methods = {
        "answer_only": plain,
        "guard": guard,
        "p_true": lambda r: (plain(r), p_true(model, tokenizer, [dict(r, answer="yes")], yes, no)),
        "self_consistency": lambda r: (plain(r), agreement(model, tokenizer, [dict(r, answer="yes")], args.samples, args.max_new_tokens)),
        "consensus": lambda r: (plain(r), pivot_answers(model, tokenizer, [r], 0, args.max_new_tokens), pivot_answers(model, tokenizer, [r], 1, args.max_new_tokens)),
    }
    report = {"n": len(picked), "gpu": torch.cuda.get_device_name(0)}
    for name, function in methods.items():
        for row in picked[: args.warmup]:
            function(row)
        times = np.array([timed(lambda: function(row))[0] for row in picked])
        report[name] = {"mean": float(times.mean()), "median": float(np.median(times)), "p95": float(np.percentile(times, 95))}
        print(name, report[name], flush=True)
    report["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 2**30
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
