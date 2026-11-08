"""Cross-lingual consensus baseline: answer in two other languages and compare."""
import argparse
import json
import os

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from baselines import chat
from build_rag import ENCODER, embed
from gate import LONG_PASSAGE_CHARS

PIVOTS = {"en": ("Hindi", "French"), "te": ("English", "Hindi"), "bn": ("English", "Hindi"), "sw": ("English", "Hindi")}
ASK = "Answer the question in {pivot}, using only a few words.\n\nPassage: {context}\n\nQuestion: {question}"


@torch.no_grad()
def answers(model, tokenizer, batch, pivot_index, max_new_tokens):
    prompts = [
        chat(tokenizer, ASK.format(pivot=PIVOTS[r["language"]][pivot_index], context=r["context"], question=r["question"]))
        for r in batch
    ]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    sequences = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.batch_decode(sequences[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="CohereLabs/tiny-aya-fire")
    parser.add_argument("--data", default="../data/questions.jsonl")
    parser.add_argument("--run", default="../runs/rag_calibration")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()

    contexts = {}
    for line in open(args.data, encoding="utf-8"):
        row = json.loads(line)
        contexts[row["id"]] = row["context"]
    records = [json.loads(line) for line in open(f"{args.run}/answers.jsonl", encoding="utf-8")]
    for r in records:
        r["context"] = contexts[r["id"]]
    records.sort(key=lambda r: -len(r["context"]))

    token = os.environ["HF_TOKEN"]
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, token=token, dtype=torch.bfloat16, device_map="cuda").eval()

    start = 0
    while start < len(records):
        size = 1 if len(records[start]["context"]) > LONG_PASSAGE_CHARS else args.batch_size
        batch = records[start : start + size]
        for index in (0, 1):
            for r, text in zip(batch, answers(model, tokenizer, batch, index, args.max_new_tokens)):
                r[f"pivot_{index}"] = text
        start += len(batch)
        if start % 200 < size:
            print(f"{start}/{len(records)}", flush=True)
    del model
    torch.cuda.empty_cache()

    encoder_tokenizer = AutoTokenizer.from_pretrained(ENCODER)
    encoder = AutoModel.from_pretrained(ENCODER).to("cuda").eval()
    own = embed([r["answer"] for r in records], encoder_tokenizer, encoder, "query: ")
    first = embed([r["pivot_0"] for r in records], encoder_tokenizer, encoder, "query: ")
    second = embed([r["pivot_1"] for r in records], encoder_tokenizer, encoder, "query: ")
    agreement = (F.cosine_similarity(own, first) + F.cosine_similarity(own, second) + F.cosine_similarity(first, second)) / 3
    with open(f"{args.run}/consensus.jsonl", "w", encoding="utf-8") as out:
        for r, value in zip(records, agreement.tolist()):
            out.write(json.dumps({"id": r["id"], "consensus": value, "pivot_0": r["pivot_0"], "pivot_1": r["pivot_1"]}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
