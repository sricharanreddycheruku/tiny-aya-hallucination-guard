# Knows When It Doesn't Know

A retrieve-first hallucination guard for Tiny Aya in English, Telugu, Bengali and Swahili.

```
question -> retrieve passage -> Tiny Aya answer + activation -> risk score -> one global threshold -> answer or abstain
```

On the locked TyDiQA test set (1,721 questions), wrong answers delivered fall from 48.6% to 6.6%, while right
answers delivered fall from 51.4% to 39.7%. The guard answers 46.3% of questions. The full write-up is in
[ARTICLE.md](ARTICLE.md).

## Layout

| Path | Contents |
|---|---|
| `src/` | Every experiment script, run from inside `src/` |
| `tests/` | Unit tests for scoring and calibration |
| `space/` | The Gradio demo |
| `notebooks/results.ipynb` | Rebuilds the result tables from `reports/` on a CPU |
| `reports/` | Final reports, frozen settings, latency, ROI and circuit overlap |
| `hf/` | Cards for the model and dataset repositories |

## Setup

Tested on Python 3.12 with one 16 GB RTX 5060 Ti. Peak GPU memory for serving is 8.5 GB.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
git clone https://github.com/Lexsi-Labs/CuratorKIT third_party/CuratorKIT
git clone https://github.com/Lexsi-Labs/CircuitKIT third_party/CircuitKIT
git clone https://github.com/Lexsi-Labs/AuditKIT third_party/AuditKIT
.venv/bin/pip install -e third_party/CuratorKIT -e third_party/CircuitKIT -e third_party/AuditKIT
export HF_TOKEN=...   # an account that has accepted the Tiny Aya licence
.venv/bin/python -m pytest tests -q
```

## Rebuild the results from the saved reports

Open `notebooks/results.ipynb`. It needs no GPU.

## Reproduce from scratch

Run from `src/`. All seeds are fixed in the script defaults. Greedy answers can still differ slightly across GPUs and batch sizes.

```bash
PY=../.venv/bin/python
G=CohereLabs/tiny-aya-global
H=../data/health_big.jsonl

# Questions and retrieved passages
$PY build_rag.py

# Answers and activations
for s in probe calibration test; do
  $PY gate.py --input ../data/questions.jsonl --split $s --batch-size 8 --out ../runs/rag_$s
  $PY gate.py --model $G --input ../data/questions.jsonl --split $s --batch-size 8 --out ../runs/global_$s
done
$PY gate.py --input ../data/questions.jsonl --split test --closed-book --batch-size 8 --out ../runs/closed_test

# Competing scores
for r in rag_calibration rag_test; do $PY baselines.py --run ../runs/$r; $PY consensus.py --run ../runs/$r; done
$PY baselines.py --model $G --run ../runs/global_test
$PY consensus.py --model $G --run ../runs/global_test

# Health domain set with CuratorKIT
$PY health_corpus.py
$PY curate.py --per-language 800 --out $H
$PY gate.py --input $H --split health --batch-size 8 --out ../runs/rag_health_big
$PY baselines.py --data $H --run ../runs/rag_health_big
$PY consensus.py --data $H --run ../runs/rag_health_big
$PY gate_check.py
$PY strict_health.py

# Circuits with CircuitKIT, on CPU in float32
mkdir -p ../runs/circuit
export CUDA_VISIBLE_DEVICES=
$PY circuit_task.py --per-language 32 --out ../runs/circuit/pairs.csv
$PY discover.py --pairs ../runs/circuit/pairs.csv --examples 128 --out ../runs/circuit/evidence.pt
$PY circuit_task.py --out ../runs/circuit/pairs64.csv
$PY -c "import pandas as p; d=p.read_csv('../runs/circuit/pairs64.csv'); [g.to_csv(f'../runs/circuit/pairs_{l}.csv', index=False) for l, g in d.groupby('language')]"
for l in en te bn sw; do
  $PY discover.py --pairs ../runs/circuit/pairs_$l.csv --examples 64 --out ../runs/circuit/evidence_$l.pt
done
$PY overlap.py
unset CUDA_VISIBLE_DEVICES

# Probes on circuit-selected layers
for s in probe calibration; do
  $PY gate.py --input ../data/questions.jsonl --split $s --batch-size 8 --layers $(seq 1 36) --out ../runs/all_$s
done
CUDA_VISIBLE_DEVICES= $PY circuit_probe.py

# Frozen guard and AuditKIT reports
$PY final.py --eval-run ../runs/rag_test --closed-run ../runs/closed_test --out ../runs/final/rag_test.json
$PY final.py --train-run ../runs/global_probe --calibration-run ../runs/global_calibration --eval-run ../runs/global_test --out ../runs/final/global_test.json
$PY final.py --data ../data/questions.jsonl $H --eval-run ../runs/rag_health_big --out ../runs/final/health_big.json
$PY final.py --data ../data/questions.jsonl $H --eval-run ../runs/rag_health_strict --out ../runs/final/health_strict.json

# Export, latency, cost, figures
$PY export_guard.py
$PY latency.py
$PY roi.py
$PY figures.py
```

Delete `src/cache/` before reusing a pairs filename, because CircuitKIT caches by filename.

## Rules the numbers depend on

- Probe 70% and calibration 30% of TyDiQA train articles, split by title hash. TyDiQA validation is the locked test set.
- Models are fit on probe and thresholds are set on calibration. Test results were read once, after the settings in `reports/FREEZE.txt` were recorded.
- Scoring in `src/scoring.py` is string matching against gold answers, with no LLM judge.

## Licences

The code in this repository is released under [LICENSE](LICENSE). Tiny Aya is CC-BY-NC-4.0, so the demo and anything derived from
its outputs are for non-commercial use. TyDiQA is Apache-2.0, multilingual-e5-small is MIT, and the health passages
come from Wikipedia under CC BY-SA 4.0.
