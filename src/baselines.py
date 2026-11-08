"""Competing risk scores: retrieval similarity, P(True) and self-consistency."""
import argparse
import json
import os

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from build_rag import ENCODER, embed
from gate import INSTRUCTION, LONG_PASSAGE_CHARS
from scoring import token_f1

CHECK = "Passage: {context}\n\nQuestion: {question}\n\nProposed answer: {answer}\n\nIs the proposed answer correct? Reply with Yes or No."


def chat(tokenizer, text):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def p_true(model, tokenizer, batch, yes, no):
    prompts = [chat(tokenizer, CHECK.format(**r)) for r in batch]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    logits = model(**enc, logits_to_keep=1).logits[:, -1].float()
    return torch.softmax(logits[:, [yes, no]], dim=-1)[:, 0].tolist()


@torch.no_grad()
def agreement(model, tokenizer, batch, samples, max_new_tokens):
    prompts = [chat(tokenizer, f"{INSTRUCTION}Passage: {r['context']}\n\nQuestion: {r['question']}") for r in batch]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    sequences = model.generate(
        **enc, do_sample=True, temperature=0.7, top_p=0.95, num_return_sequences=samples,
        max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
    )
    texts = tokenizer.batch_decode(sequences[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    values = []
    for i, r in enumerate(batch):
        drawn = texts[i * samples : (i + 1) * samples]
        values.append(sum(token_f1(t, [r["answer"]], r["language"]) >= 0.5 for t in drawn) / samples)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="CohereLabs/tiny-aya-fire")
    parser.add_argument("--data", default="../data/questions.jsonl")
    parser.add_argument("--run", default="../runs/rag_calibration")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()

    contexts = {}
    for line in open(args.data, encoding="utf-8"):
        row = json.loads(line)
        contexts[row["id"]] = row["context"]
    records = [json.loads(line) for line in open(f"{args.run}/answers.jsonl", encoding="utf-8")]
    for r in records:
        r["context"] = contexts[r["id"]]

    encoder_tokenizer = AutoTokenizer.from_pretrained(ENCODER)
    encoder = AutoModel.from_pretrained(ENCODER).to("cuda").eval()
    questions = embed([r["question"] for r in records], encoder_tokenizer, encoder, "query: ")
    passages = embed([r["context"] for r in records], encoder_tokenizer, encoder, "passage: ")
    similarity = F.cosine_similarity(questions, passages).tolist()
    del encoder, questions, passages
    torch.cuda.empty_cache()

    token = os.environ["HF_TOKEN"]
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, token=token, dtype=torch.bfloat16, device_map="cuda").eval()
    yes = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no = tokenizer.encode("No", add_special_tokens=False)[0]

    with open(f"{args.run}/baselines.jsonl", "w", encoding="utf-8") as out:
        start = 0
        while start < len(records):
            size = 1 if len(records[start]["context"]) > LONG_PASSAGE_CHARS else args.batch_size
            batch = records[start : start + size]
            trust = p_true(model, tokenizer, batch, yes, no)
            agree = agreement(model, tokenizer, batch, args.samples, args.max_new_tokens)
            for i, r in enumerate(batch):
                out.write(json.dumps({
                    "id": r["id"], "similarity": similarity[start + i], "p_true": trust[i], "agreement": agree[i],
                }) + "\n")
            out.flush()
            start += len(batch)
            print(f"{start}/{len(records)}", flush=True)


if __name__ == "__main__":
    main()
