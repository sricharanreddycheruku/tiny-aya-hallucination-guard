"""Retrieve, answer, read the layer-24 activation, score the risk, then answer or abstain."""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from guard import LAYER, Guard

MODEL = "CohereLabs/tiny-aya-fire"
ENCODER = "intfloat/multilingual-e5-small"
INSTRUCTION = "Answer the question in the same language, using only a few words.\n\n"
MAX_NEW_TOKENS = 24


class Pipeline:
    def __init__(self, folder=".", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        token = os.environ.get("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, token=token, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, token=token, dtype=torch.bfloat16).to(self.device).eval()
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        self.stop_ids = torch.tensor(sorted(set(eos + [self.tokenizer.pad_token_id])), device=self.device)
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(ENCODER)
        self.encoder = AutoModel.from_pretrained(ENCODER).to(self.device).eval()
        self.guard = Guard(f"{folder}/guard_fire.npz")
        self.corpus = [json.loads(line) for line in open(f"{folder}/health_chunks.jsonl", encoding="utf-8")]
        self.vectors = torch.tensor(np.load(f"{folder}/health_vectors.npy"), device=self.device)
        self.languages = np.array([row["language"] for row in self.corpus])

    @torch.no_grad()
    def embed(self, text, prefix):
        enc = self.encoder_tokenizer(
            [prefix + text], padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(self.device)
        hidden = self.encoder(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        return F.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1)[0]

    def retrieve(self, question, language):
        scores = self.vectors @ self.embed(question, "query: ")
        scores[torch.tensor(self.languages != language, device=self.device)] = -1.0
        best = int(scores.argmax())
        return self.corpus[best]

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

    def respond(self, question, language, passage=None, rule="joint"):
        passage = passage or self.retrieve(question, language)
        answer, activation, mean_logprob = self.generate(question, passage["text"])
        similarity = float(self.embed(question, "query: ") @ self.embed(passage["text"], "passage: "))
        risk = float(self.guard.risk(activation[None], np.array([similarity]), np.array([mean_logprob]))[0])
        return {
            "answer": answer, "risk": risk, "similarity": similarity, "mean_logprob": mean_logprob,
            "threshold": float(self.guard.w[f"threshold_{rule}"]),
            "delivered": bool(answer) and bool(self.guard.decide(risk, rule)), "passage": passage,
        }
