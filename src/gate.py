"""Closed-book TyDiQA run: answers, correctness, confidence and activations."""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from scoring import exact_match, is_wrong, normalize, token_f1

LANGUAGES = {"english": "en", "telugu": "te", "bengali": "bn", "swahili": "sw"}
INSTRUCTION = "Answer the question in the same language, using only a few words.\n\n"
LAYERS = list(range(4, 37, 4))
LONG_PASSAGE_CHARS = 3000


def load_questions(per_language, seed):
    data = load_dataset("google-research-datasets/tydiqa", "secondary_task", split="train")
    rng = random.Random(seed)
    grouped = {}
    for row in data:
        name = row["id"].split("-")[0]
        if name in LANGUAGES:
            grouped.setdefault(LANGUAGES[name], []).append(row)
    picked = []
    for language, rows in grouped.items():
        rng.shuffle(rows)
        for row in rows[:per_language]:
            picked.append({
                "id": row["id"],
                "language": language,
                "title": row["title"],
                "question": row["question"],
                "context": row["context"],
                "golds": sorted(set(row["answers"]["text"])),
            })
    return picked


@torch.no_grad()
def run_batch(model, tokenizer, batch, max_new_tokens, stop_ids, open_book=False, layers=LAYERS):
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": INSTRUCTION + (f"Passage: {item['context']}\n\nQuestion: " if open_book else "") + item["question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for item in batch
    ]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    prompt_len = enc["input_ids"].shape[1]
    sequences = model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id
    )
    answer_ids = sequences[:, prompt_len:]
    answer_mask = ~torch.isin(answer_ids, stop_ids)
    full_mask = torch.cat([enc["attention_mask"], answer_mask.long()], dim=1)
    out = model(
        input_ids=sequences,
        attention_mask=full_mask,
        output_hidden_states=True,
        logits_to_keep=answer_ids.shape[1] + 1,
    )

    logits = out.logits[:, :-1].float()
    token_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, answer_ids.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs.masked_fill(~answer_mask, 0.0)
    lengths = answer_mask.sum(dim=1)
    mean_logprob = token_logprobs.sum(dim=1) / lengths.clamp(min=1)
    min_logprob = token_logprobs.masked_fill(~answer_mask, 0.0).min(dim=1).values

    last = prompt_len + (lengths - 1).clamp(min=0)
    rows = torch.arange(len(batch), device=sequences.device)
    prompt_acts = torch.stack([out.hidden_states[l][:, prompt_len - 1] for l in layers], dim=1)
    answer_acts = torch.stack([out.hidden_states[l][rows, last] for l in layers], dim=1)
    answers = tokenizer.batch_decode(answer_ids.masked_fill(~answer_mask, tokenizer.pad_token_id), skip_special_tokens=True)
    return (
        [a.strip() for a in answers],
        mean_logprob.cpu().numpy(),
        min_logprob.cpu().numpy(),
        lengths.cpu().numpy(),
        prompt_acts.float().cpu().numpy().astype(np.float16),
        answer_acts.float().cpu().numpy().astype(np.float16),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="CohereLabs/tiny-aya-fire")
    parser.add_argument("--per-language", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--open-book", action="store_true")
    parser.add_argument("--closed-book", action="store_true", help="ignore the context of --input rows")
    parser.add_argument("--input", help="questions file with a context field per row")
    parser.add_argument("--split", help="row split to run from --input")
    parser.add_argument("--layers", type=int, nargs="+", default=LAYERS, help="hidden state indices to save")
    parser.add_argument("--out", default="runs/gate_fire")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ["HF_TOKEN"]
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, token=token, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    stop_ids = torch.tensor(sorted(set(eos + [tokenizer.pad_token_id])), device=model.device)

    if args.input:
        rows = [json.loads(line) for line in open(args.input, encoding="utf-8")]
        items = sorted((r for r in rows if r["split"] == args.split), key=lambda r: -len(r["context"]))
        args.open_book = not args.closed_book
    else:
        items = load_questions(args.per_language, args.seed)
    prompt_acts, answer_acts = [], []
    with open(out_dir / "answers.jsonl", "w", encoding="utf-8") as log:
        start = 0
        while start < len(items):
            long_passage = len(items[start].get("context", "")) > LONG_PASSAGE_CHARS
            batch = items[start : start + (1 if long_passage else args.batch_size)]
            answers, mean_lp, min_lp, lengths, p_act, a_act = run_batch(
                model, tokenizer, batch, args.max_new_tokens, stop_ids, args.open_book, args.layers
            )
            prompt_acts.append(p_act)
            answer_acts.append(a_act)
            for i, item in enumerate(batch):
                language = item["language"]
                record = dict(
                    {k: v for k, v in item.items() if k != "context"},
                    answer=answers[i],
                    mean_logprob=float(mean_lp[i]),
                    min_logprob=float(min_lp[i]),
                    n_tokens=int(lengths[i]),
                    f1=token_f1(answers[i], item["golds"], language),
                    exact=exact_match(answers[i], item["golds"], language),
                    contains=any(normalize(g, language) in normalize(answers[i], language) for g in item["golds"]),
                    wrong=is_wrong(answers[i], item["golds"], language),
                )
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()
            start += len(batch)
            print(f"{start}/{len(items)}", flush=True)
    np.savez(
        out_dir / "activations.npz",
        prompt=np.concatenate(prompt_acts),
        answer=np.concatenate(answer_acts),
        layers=np.array(args.layers),
    )


if __name__ == "__main__":
    main()
