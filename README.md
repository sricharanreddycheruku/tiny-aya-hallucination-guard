# Knows When It Doesn't Know

A retrieve-first hallucination guard for Tiny Aya in English, Telugu, Bengali and Swahili. Built for Lexsi Track 2b, Problem 13.

## The activation probe answers most questions

On 1,721 TyDiQA test questions, the probe answered **55.1%** and delivered **76.2% fewer wrong answers** than RAG answering every question.

| Policy | Coverage | Correct delivered | Wrong delivered |
|---|---:|---:|---:|
| RAG always answers | 100.0% | 881 | 840 |
| Published guard replay | 45.8% | 679 | 110 |
| **Activation probe, 55% target** | **55.1%** | **748** | **200** |
| No activation, 55% target | 57.5% | 689 | 301 |

The probe delivered 59 more correct answers and 101 fewer wrong answers than the no-activation baseline at similar coverage.
Compared with the published guard replay, it delivered 69 more correct answers and 90 more wrong answers.
See the [full follow-up results](#tiny-aya-fire-follow-up-results) and [submission abstract](ABSTRACT.md).

CuratorKIT built the health benchmark, CircuitKIT tested which activations carried evidence signals, and AuditKIT produced paired comparisons.

```
question -> retrieve passage -> Tiny Aya answer + activation -> risk score -> one global threshold -> answer or abstain
```

## Original locked TyDiQA results

| Metric (1,721 questions) | RAG always answers | RAG with guard |
|---|---:|---:|
| Questions answered | 100% | 46.3% |
| Wrong delivered, out of all questions | 48.6% | 6.6% |
| Right delivered, out of all questions | 51.4% | 39.7% |
| Wrong among delivered answers | 48.6% | 14.2% |

The guard withheld 724 wrong answers and 201 right answers. See the [write-up](ARTICLE.md), [abstract](ABSTRACT.md),
[model collection](https://huggingface.co/collections/sricharan1234/tiny-aya-hallucination-guard-models-6af0a41dfe3d3d356a5000db)
and [dataset collection](https://huggingface.co/collections/sricharan1234/tiny-aya-hallucination-guard-datasets-6af0a41ef264c137a04554d2).

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
git -C third_party/CuratorKIT checkout --detach dd9e835301fcddc74a37501d9e207a5a060d209f
git -C third_party/CircuitKIT checkout --detach e71d5e17fe6e2abfde47ec9a7fbaf31649601b78
git -C third_party/AuditKIT checkout --detach 71316da345784ff220d5ed59337441a59ab31526
.venv/bin/pip install -e third_party/CuratorKIT -e third_party/CircuitKIT -e third_party/AuditKIT
export HF_TOKEN=...   # an account that has accepted the Tiny Aya licence
.venv/bin/python -m pytest tests -q
```

The toolkit pins use the last first-parent commits available before the recorded experiment freeze. The original write-up
did not record upstream SHAs. CuratorKIT builds the health benchmark, CircuitKIT runs circuit discovery and probe comparisons,
and AuditKIT produces the paired reports.

## Run the demo from a fresh checkout

Use Python 3.12 and a PyTorch environment with CUDA for the GPU path. Run the commands at the repository root.
Set `HF_TOKEN` locally with read access from an account that has accepted the
[Tiny Aya Fire model terms](https://huggingface.co/CohereLabs/tiny-aya-fire).

```bash
python -m pip install -r space/requirements.txt
python src/build_space.py --device cuda
export AYA_ASSETS_DIR=space_build
python space/app.py
```

In PowerShell, set the asset directory with `$env:AYA_ASSETS_DIR = 'space_build'`. The builder downloads the pinned
public guard and health corpus and indexes the passages. It accepts `--device cpu` for indexing and
`--guard` / `--chunks` for local inputs. The optional second-passage retry is experimental and off by default.

## Rebuild the results from the saved reports

Open `notebooks/results.ipynb`. It needs no GPU.

## Reproduce from scratch

Run from `src/`. All seeds are fixed in the script defaults. Greedy answers can still differ slightly across GPUs and batch sizes.
The published reports used the scorer at [65bf238](https://github.com/sricharanreddycheruku/tiny-aya-hallucination-guard/tree/65bf2381cc006d362488f050d0ebb529f3e51e7c).
Use that revision to replay the original scorer. A new run with the current scorer produces revised automatic labels.

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

## Follow-up evaluation

After generating the probe, calibration and test activation files, run these commands from the repository root.
The saved Hugging Face answer logs can be rescored on CPU. They do not contain the activation arrays.

```bash
python src/rescore.py runs/rag_test/answers.jsonl --summary-only
python src/build_space.py --fetch-only
python src/followup.py
python src/evaluate_retry.py --questions data/health_big.jsonl --assets space_build
```

`followup.py` fits the full guard, activation-only and no-activation comparators on revised labels. It chooses thresholds
for 55% and 60% coverage on calibration scores, then reports actual test coverage, wrong and right delivery rates, and
error among delivered answers. Its report includes pooled, per-language, source-exclusion and gold-string slices. The
risk-coverage plot is a descriptive test-set comparison. The independent-pilot experiment separates articles when selecting
candidate thresholds and estimating risk. Its binomial bound assumes independent certification rows. Repeated questions
within an article can violate that assumption.

The report also replays the published joint guard on the same candidate activations and shows results under both the
stored and revised labels. Use the revised-label replay for comparisons with the follow-up policies. The retry benchmark
uses a built passage index. Run the demo preparation command without `--fetch-only` before benchmarking retries.

An offline rescore of the public answer logs changed 7 of 1,721 TyDiQA labels (837 to 840 marked wrong) and 4 of 1,173
health labels (669 to 667 marked wrong). These are changes to automatic labels on identical generated answers.

### Tiny Aya Fire follow-up results

We regenerated the three splits on an RTX 5060 Ti (BF16, batch size 4) and evaluated every policy on the same 1,721 test
questions with the revised automatic scorer. This is a follow-up to the original locked benchmark.

| Policy | Coverage | Correct delivered | Wrong delivered | Wrong among delivered |
|---|---:|---:|---:|---:|
| RAG always answers | 100.0% | 881 | 840 | 48.8% |
| Published joint guard, replayed | 45.8% | 679 | 110 | 13.9% |
| Refitted guard, 10% joint calibration target | 44.5% | 666 | 99 | 12.9% |
| Refitted guard, 55% coverage target | 55.6% | 745 | 212 | 22.2% |
| Probe only, 55% coverage target | 55.1% | 748 | 200 | 21.1% |
| No activation, 55% coverage target | 57.5% | 689 | 301 | 30.4% |
| Refitted guard, 60% coverage target | 62.2% | 785 | 285 | 26.6% |

The 55% target guard delivers 628 fewer wrong answers than always answering and retains majority coverage overall.
Bengali coverage is 46.0% at this setting. Probe-only delivers three more correct answers and twelve fewer wrong answers
than the full guard at a similar overall coverage. The refitted guard at the original joint calibration target is more
conservative than the published replay. We report these as observed trade-offs. The retry policy and multilingual human
review are pending.

See the [majority-coverage report](reports/followup/group12-guard-benchmark-report.json) and
[joint-risk comparison](reports/followup/group12-matched-risk.json). For the latter, the new thresholds use
`conformal_threshold` with alpha 0.10 on the calibration split. The saved majority report's `stored_labels` field refers
to the labels in the regenerated answer log, which already used the revised scorer.

### Health-domain recalibration

From the repository root, use the health questions in the linked dataset collection and the published guard weights.

```bash
python src/gate.py --input data/health_big.jsonl --split health --batch-size 4 --out runs/rag_health_recalibrate
python src/health_recalibrate.py --data data/health_big.jsonl --run runs/rag_health_recalibrate --guard space_build/guard_fire.npz --out runs/followup/health-recalibration.json
```

The follow-up keeps the guard weights fixed and compares its published threshold with pooled and per-language thresholds
calibrated for a 10% wrong-delivery target on 40% of topics. It evaluates all three policies on the same held-out topics
with the revised automatic scorer. Paired topic-bootstrap intervals describe uncertainty in differences from the
published threshold. The target is a calibration objective, and test performance may differ.

Fill `factually_correct` and `supported_by_passage` independently in `runs/followup/review.csv` using `yes`, `no` or
`unclear`. Then run `python src/review_agreement.py runs/followup/review.csv runs/rag_test/answers.jsonl` to compare the
old stored labels and the revised scorer with the factual reviews. For that comparison, use the original saved answer
log with its original labels. For a demo recording, show one correct delivery, one wrong candidate withheld, and one
correct candidate withheld, using completed reviews to establish their outcomes.

## Rules the numbers depend on

- Probe 70% and calibration 30% of TyDiQA train articles, split by title hash. TyDiQA validation is the locked test set.
- Models are fit on probe and thresholds are set on calibration. Test results were read once, after the settings in `reports/FREEZE.txt` were recorded.
- Scoring in `src/scoring.py` is string matching against gold answers, with no LLM judge.

## Licences

The code in this repository is released under [LICENSE](LICENSE). Tiny Aya is CC-BY-NC-4.0, so the demo and anything derived from
its outputs are for non-commercial use. TyDiQA is Apache-2.0, multilingual-e5-small is MIT, and the health passages
come from Wikipedia under CC BY-SA 4.0.
