"""Retrieve, answer, read the layer-24 activation, score the risk, then answer or abstain."""
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from guard import LAYER, Guard

MODEL = "CohereLabs/tiny-aya-fire"
ENCODER = "intfloat/multilingual-e5-small"
INSTRUCTION = "Answer the question in the same language, using only a few words.\n\n"
MAX_NEW_TOKENS = 24


class Pipeline:
    def __init__(self, folder=".", device=None):
        folder = Path(folder)
        required = ("guard_fire.npz", "health_chunks.jsonl", "health_vectors.npy")
        missing = [name for name in required if not (folder / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing demo assets in {folder.resolve()}: {', '.join(missing)}. Run src/build_space.py.")
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        token = os.environ.get("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, token=token, padding_side="left")
        dtype = torch.float32 if torch.device(self.device).type == "cpu" else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, token=token, dtype=dtype).to(self.device).eval()
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        self.stop_ids = torch.tensor(sorted(set(eos + [self.tokenizer.pad_token_id])), device=self.device)
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(ENCODER)
        self.encoder = AutoModel.from_pretrained(ENCODER).to(self.device).eval()
        self.guard = Guard(folder / "guard_fire.npz")
        with (folder / "health_chunks.jsonl").open(encoding="utf-8") as handle:
            self.corpus = [json.loads(line) for line in handle]
        self.vectors = torch.as_tensor(np.load(folder / "health_vectors.npy"), device=self.device)
        if self.vectors.ndim != 2 or len(self.vectors) != len(self.corpus):
            raise ValueError("health_vectors.npy must contain one vector per corpus row.")
        self.languages = np.array([row["language"] for row in self.corpus])
        self.topics = np.array([row["topic"] for row in self.corpus])

    @torch.no_grad()
    def embed(self, text, prefix):
        enc = self.encoder_tokenizer(
            [prefix + text], padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(self.device)
        hidden = self.encoder(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        return F.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1)[0]

    def retrieve(self, question, language):
        return self.rank_passages(self.embed(question, "query: "), language)[0][0]

    def rank_passages(self, query, language, limit=1, excluded_topic=None):
        eligible = np.flatnonzero((self.languages == language) & (self.topics != excluded_topic))
        if not len(eligible):
            raise ValueError(f"No eligible passages for {language!r}.")
        indices = torch.as_tensor(eligible, device=self.device)
        scores = self.vectors[indices] @ query
        if limit == 1:
            offset = int(scores.argmax())
            return [(self.corpus[eligible[offset]], float(scores[offset]))]
        order = torch.argsort(scores, descending=True, stable=True).tolist()
        passages = []
        seen_text = set()
        for offset in order:
            passage = self.corpus[eligible[offset]]
            if passage["text"] in seen_text:
                continue
            seen_text.add(passage["text"])
            passages.append((passage, float(scores[offset])))
            if len(passages) == limit:
                break
        return passages

    @torch.no_grad()
    def generate(self, question, context):
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{INSTRUCTION}Passage: {context}\n\nQuestion: {question}"}],
            tokenize=False, add_generation_prompt=True,
        )
        enc = self.tokenizer([prompt], return_tensors="pt", add_special_tokens=False).to(self.device)
        prompt_len = enc["input_ids"].shape[1]
        sequences = self.model.generate(
            **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=self.tokenizer.pad_token_id
        )
        answer_ids = sequences[:, prompt_len:]
        answer_mask = ~torch.isin(answer_ids, self.stop_ids)
        out = self.model(
            input_ids=sequences,
            attention_mask=torch.cat([enc["attention_mask"], answer_mask.long()], dim=1),
            output_hidden_states=True,
            logits_to_keep=answer_ids.shape[1] + 1,
        )
        logprobs = torch.log_softmax(out.logits[:, :-1].float(), dim=-1)
        logprobs = logprobs.gather(-1, answer_ids.unsqueeze(-1)).squeeze(-1).masked_fill(~answer_mask, 0.0)
        length = int(answer_mask.sum())
        mean_logprob = float(logprobs.sum()) / max(length, 1)
        activation = out.hidden_states[LAYER][0, prompt_len + max(length - 1, 0)].float().cpu().numpy()
        answer = self.tokenizer.decode(
            answer_ids.masked_fill(~answer_mask, self.tokenizer.pad_token_id)[0], skip_special_tokens=True
        ).strip()
        return answer, activation.astype(np.float16).astype(np.float32), mean_logprob

    def score_passage(self, question, passage, similarity, rule):
        answer, activation, mean_logprob = self.generate(question, passage["text"])
        risk = float(self.guard.risk(activation[None], np.array([similarity]), np.array([mean_logprob]))[0])
        return {
            "answer": answer, "risk": risk, "similarity": similarity, "mean_logprob": mean_logprob,
            "threshold": float(self.guard.w[f"threshold_{rule}"]),
            "delivered": bool(answer) and bool(self.guard.decide(risk, rule)), "passage": passage,
        }

    def respond(self, question, language, passage=None, rule="joint", retry=False, excluded_topic=None):
        query = self.embed(question, "query: ")
        if passage is not None:
            similarity = float(query @ self.embed(passage["text"], "passage: "))
            return self.score_passage(question, passage, similarity, rule)

        ranked = self.rank_passages(query, language, limit=2 if retry else 1, excluded_topic=excluded_topic)
        first_passage, first_similarity = ranked[0]
        first = self.score_passage(question, first_passage, first_similarity, rule)
        if first["delivered"] or len(ranked) == 1:
            return first
        second_passage, second_similarity = ranked[1]
        started = perf_counter()
        second = self.score_passage(question, second_passage, second_similarity, rule)
        return dict(second, first_attempt=first, retrieval_attempts=2, retry_seconds=perf_counter() - started)
