---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-fire
language: [en, te, bn, sw]
tags: [hallucination-detection, selective-prediction, probe, multilingual]
---

# Tiny Aya hallucination guard

A 27 KB linear guard for `CohereLabs/tiny-aya-fire`. It estimates the chance that an open-book answer is wrong
from three inputs: the layer-24 hidden state at the last answer token, the question-passage cosine similarity from
`intfloat/multilingual-e5-small`, and the mean answer log-probability.

This repository holds the guard only. The base model is gated and is not redistributed.

## Files

| File | Contents |
|---|---|
| `guard_fire.npz` | Scaler, probe and stacker weights, and the two frozen thresholds |
| `guard.py` | Loads the weights and returns a risk score and a decision |
| `detector_probes.pt` | CircuitKIT `LinearProbe` weights from the circuit comparison, kept for reference, not used by the guard |

## Use

```python
from guard import Guard

guard = Guard("guard_fire.npz")
risk = guard.risk(activation, similarity, mean_logprob)  # arrays with one row per question
deliver = guard.decide(risk, rule="joint")               # or rule="selective"
```

The activation must come from `hidden_states[24]` at the last non-stop answer token, using the prompt and decoding
settings in `space/pipeline.py`.

## Thresholds

| Rule | Risk limit | Target | Result on the locked test set |
|---|---|---|---|
| `joint` | 0.4473 | TyDiQA target of 10% wrong out of all questions | 6.6%, with 46.3% answered |
| `selective` | 0.3632 | TyDiQA target of 20% wrong among delivered answers | 10.9%, with 39.9% answered |

## Limits

- The guard is fitted to this model, prompt and layer. It does not transfer to another model without refitting.
- It was calibrated on TyDiQA, and the targets are not guaranteed under a domain change. See the write-up for health results.
- It withholds many correct answers. It lowers risk and does not make the model more accurate.
- It is not a medical device and is not for clinical decisions.

## Licence

Weights are derived from Tiny Aya activations and released under CC-BY-NC-4.0.
