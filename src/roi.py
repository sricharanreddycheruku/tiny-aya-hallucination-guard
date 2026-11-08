"""Cost of serving the guarded model against closed-model APIs, per question and per month.

Prices are USD, read from the providers' pricing pages on 2026-11-05. Tokens are counted
with the Tiny Aya tokenizer, so API token counts are an approximation.
"""
import json
import os

import numpy as np
from transformers import AutoTokenizer

from gate import INSTRUCTION

GPU_PER_HOUR = 0.80  # Hugging Face, 1x Nvidia L4
HOURS_PER_MONTH = 730
APIS = {  # input, output per million tokens
    "gpt-6-luna": (0.10, 0.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.8-flash": (0.75, 3.75),
    "gpt-6.1-sol": (2.00, 10.00),
    "gpt-6-astra": (10.00, 50.00),
}
VOLUMES = [100_000, 1_000_000, 5_000_000]


def main():
    tokenizer = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-fire", token=os.environ.get("HF_TOKEN"))
    contexts = {}
    for line in open("../data/questions.jsonl", encoding="utf-8"):
        row = json.loads(line)
        contexts[row["id"]] = row["context"]
    rows = [json.loads(line) for line in open("../runs/rag_calibration/answers.jsonl", encoding="utf-8")]
    tokens = {}
    for row in rows:
        content = f"{INSTRUCTION}Passage: {contexts[row['id']]}\n\nQuestion: {row['question']}"
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True, return_dict=True)
        pair = (len(prompt["input_ids"]), len(tokenizer(row["answer"], add_special_tokens=False)["input_ids"]))
        tokens.setdefault(row["language"], []).append(pair)
    per_language = {k: np.array(v).mean(0).tolist() for k, v in tokens.items()}
    prompt_tokens, answer_tokens = np.concatenate([np.array(v) for v in tokens.values()]).mean(0)

    seconds = json.load(open("../runs/latency.json"))["guard"]["mean"]
    capacity = HOURS_PER_MONTH * 3600 / seconds
    api_cost = {name: (prompt_tokens * i + answer_tokens * o) / 1e6 for name, (i, o) in APIS.items()}
    fixed = GPU_PER_HOUR * HOURS_PER_MONTH
    report = {
        "prompt_tokens": prompt_tokens, "answer_tokens": answer_tokens, "tokens_by_language": per_language,
        "guard_seconds": seconds, "gpu_per_hour": GPU_PER_HOUR,
        "self_host_per_1k_at_full_load": 1000 * seconds / 3600 * GPU_PER_HOUR,
        "always_on_month": fixed, "capacity_per_month": capacity,
        "api_per_1k": {k: 1000 * v for k, v in api_cost.items()},
        "break_even_per_month": {k: fixed / v for k, v in api_cost.items()},
        "month_cost": {
            str(volume): {"self_host": fixed * int(np.ceil(volume / capacity)), **{k: v * volume for k, v in api_cost.items()}}
            for volume in VOLUMES
        },
    }
    json.dump(report, open("../runs/roi.json", "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
