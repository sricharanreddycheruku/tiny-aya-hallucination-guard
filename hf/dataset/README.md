---
license: cc-by-sa-4.0
language: [en, te, bn, sw]
task_categories: [question-answering]
tags: [hallucination-detection, multilingual, health]
configs:
- config_name: health
  data_files: data/health_big.jsonl
- config_name: health_passages
  data_files: data/health_chunks.jsonl
---

# Tiny Aya hallucination guard: data and logs

Data, model answers, scores and reports behind the guard experiments.

## Contents

| Path | Rows | Contents |
|---|---|---|
| `data/questions.jsonl` | 10,521 | TyDiQA questions with retrieved passages and probe, calibration or test split labels |
| `data/health_chunks.jsonl` | 3,910 | Wikipedia passages on 40 health topics |
| `data/health_big.jsonl` | 1,173 | Questions generated with CuratorKIT |
| `runs/*/answers.jsonl` | | Tiny Aya answers, log-probabilities and scoring for each run |
| `runs/*/baselines.jsonl`, `consensus.jsonl` | | Competing scores |
| `runs/rag_health_strict/grounding.json` | 1,173 | Judge scores for the stricter health subset |
| `reports/` | | AuditKIT reports, frozen settings, latency, ROI, circuit overlap |

Activation arrays are not included. They can be rebuilt with `src/gate.py`.

## How the health set was made

Passages come from Wikipedia in each language. Tiny Aya Fire generated each question and answer, and Tiny Aya Global
judged grounding through CuratorKIT's hallucination gate. Gold answers are copied from the source text. The stricter
subset keeps gold answers scoring at least 0.85, a bar chosen with planted wrong answers. Selection uses no output
from the model under test. No human verified these questions.

## Limits

- Generator and judge belong to the same model family, so errors may be shared.
- Passage retrieval excludes the source article for approximately 30% of questions. A different retrieved article can still contain the gold answer.
- Scores are string matches, which can mark a valid paraphrase wrong.
- Bengali test size is small (113).

## Licences

- `data/questions.jsonl` derives from TyDiQA (Apache-2.0), and its passages are Wikipedia text (CC BY-SA).
- Health passages and questions derive from Wikipedia under CC BY-SA 4.0.
- Model answers and generated questions come from Tiny Aya (CC-BY-NC-4.0), so they are for non-commercial use.
