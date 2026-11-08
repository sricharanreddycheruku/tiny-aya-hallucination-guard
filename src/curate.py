"""Grounded health QA set: CuratorKIT generation and hallucination gate on local models."""
import argparse
import gc
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from curatorkit.gates.hallucination import HallucinationGate
from curatorkit.generators.qa_generator import QAGenerationTask
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from build_rag import ENCODER, embed
from scoring import normalize

NAMES = {"en": "English", "te": "Telugu", "bn": "Bengali", "sw": "Swahili"}
QA_PROMPT = """Read the text and write exactly {{num_questions}} question-answer pair(s) in {language}.
Each answer must be a short phrase copied exactly from the text, at most six words.
Each question must make sense without seeing the text, so name the disease or topic it is about.

Text:
{{context}}

Respond with JSON only:
[{{{{"question": "...", "answer": "..."}}}}]"""


class LocalLLM(BaseLLM):
    """Runs a Hugging Face chat model on the local GPU, so no paid API is needed."""

    def __init__(self, name, max_tokens=256, temperature=0.0):
        super().__init__(model=name, temperature=temperature, max_tokens=max_tokens, max_retries=0)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.network = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()

    @torch.no_grad()
    def _call(self, messages, **kwargs):
        started = time.time()
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.network.device)
        sampling = {"do_sample": True, "temperature": self.temperature, "top_p": 0.95} if self.temperature else {"do_sample": False}
        output = self.network.generate(
            **enc, **sampling, max_new_tokens=self.max_tokens, pad_token_id=self.tokenizer.pad_token_id
        )
        new = output[0, enc["input_ids"].shape[1]:]
        return LLMResponse(
            text=self.tokenizer.decode(new, skip_special_tokens=True), model=self.model,
            prompt_tokens=enc["input_ids"].shape[1], completion_tokens=len(new),
            total_tokens=enc["input_ids"].shape[1] + len(new), latency_seconds=time.time() - started,
        )

    def close(self):
        del self.network
        gc.collect()
        torch.cuda.empty_cache()


def verbatim(sample, language):
    answer, question = sample.output, sample.instruction
    if not 0 < len(answer.split()) <= 8 or answer not in sample.input:
        return False
    return normalize(answer, language) not in normalize(question, language)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="../data/health_chunks.jsonl")
    parser.add_argument("--generator", default="CohereLabs/tiny-aya-fire")
    parser.add_argument("--judge", default="CohereLabs/tiny-aya-global")
    parser.add_argument("--per-language", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--uncovered", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0, help="above 0 samples new questions on each seed")
    parser.add_argument("--out", default="../data/health_questions.jsonl")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    grouped = defaultdict(list)
    for line in open(args.chunks, encoding="utf-8"):
        row = json.loads(line)
        grouped[row["language"]].append(row)

    llm = LocalLLM(args.generator, temperature=args.temperature)
    drafts = {}
    for language, rows in grouped.items():
        picked = rng.sample(rows, min(args.per_language, len(rows)))
        samples = [
            DataSample(
                id=f"{language}-{r['topic']}-{r['chunk']}", source_uri=r["url"] or "wikipedia", input=r["text"],
                instruction="", task_type="language_modeling",
                metadata={"domain": language, "source_file": r["topic"], "chunk_index": r["chunk"]},
            )
            for r in picked
        ]
        task = QAGenerationTask(llm=llm, prompt_template=QA_PROMPT.format(language=NAMES[language]), num_questions=2)
        generated = task.run(samples)
        drafts[language] = [s for s in generated if verbatim(s, language)]
        print(language, "chunks", len(samples), "generated", len(generated), "verbatim", len(drafts[language]), flush=True)
    llm.close()

    judge = LocalLLM(args.judge, max_tokens=96)
    gate = HallucinationGate(llm=judge, threshold=args.threshold)
    kept = {}
    for language, samples in drafts.items():
        passed, rejected = gate.run(samples)
        kept[language] = passed
        print(language, "gate passed", len(passed), "rejected", len(rejected), flush=True)
    judge.close()

    tokenizer = AutoTokenizer.from_pretrained(ENCODER)
    encoder = AutoModel.from_pretrained(ENCODER).to("cuda").eval()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out:
        for language, samples in kept.items():
            passages = grouped[language]
            vectors = embed([p["text"] for p in passages], tokenizer, encoder, "passage: ")
            topics = [p["topic"] for p in passages]
            seen = set()
            samples = [s for s in samples if not (s.instruction in seen or seen.add(s.instruction))]
            queries = embed([s.instruction for s in samples], tokenizer, encoder, "query: ")
            for i, (sample, vector) in enumerate(zip(samples, queries)):
                topic = sample.metadata["source_file"]
                covered = rng.random() >= args.uncovered
                scores = vectors @ vector
                if not covered:
                    scores[[j for j, t in enumerate(topics) if t == topic]] = -1.0
                best = int(scores.argmax())
                context = passages[best]["text"]
                out.write(json.dumps({
                    "id": f"health-{language}-{i}", "language": language, "title": topic,
                    "question": sample.instruction, "golds": [sample.output], "context": context,
                    "source_text": sample.input, "split": "health", "covered": covered,
                    "answer_in_context": normalize(sample.output, language) in normalize(context, language),
                    "same_article": topics[best] == topic,
                }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
