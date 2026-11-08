# Knows When It Doesn't Know: a hallucination guard for Tiny Aya in four languages

## Problem

Small multilingual models are cheap enough to serve communities that large commercial models reach poorly, but they are often wrong and sound equally sure either way. Asked from memory, Tiny Aya answers 94.8% of our Telugu test questions wrongly and 72.7% of the English ones. In a setting such as health, a confident wrong answer does more harm than no answer. We built a guard that lets the system say it does not know in English, Telugu, Bengali and Swahili.

## Method

The pipeline is: question, retrieve a passage, Tiny Aya answers while we read its layer-24 hidden state at the last answer token, compute a risk score, apply one global threshold, then answer or abstain. The risk score stacks a linear probe on that activation with retrieval similarity and mean log-probability. The probe and stacker are fitted on a probe split, and the threshold is set on a separate calibration split. Scoring is string matching against gold answers, with no LLM judge. The layer, thresholds and risk targets were frozen, with file hashes, before any test result was read.

## Results

On the locked TyDiQA test set of 1,721 questions, answering everything delivers wrong answers on 48.6% of questions. With the guard, 6.6% of questions receive a wrong answer (a 42.07-point drop, 95% interval 39.80 to 44.39). Right answers delivered fall from 51.4% to 39.7%. One global threshold keeps every language between 5.9% and 7.4%, under the 10% target.

Thresholding mean log-probability, a common alternative, fails under one global threshold: it delivers wrong answers on 26.9% of Telugu questions, where the guard delivers 6.3%. Confidence means different things across languages, while the activation-based guard stays within target under the same threshold. The method also holds on a second model, Tiny Aya Global, with its own refitted guard: wrong answers delivered fall from 49.0% to 6.7%.

## Domain shift to health

We applied the frozen thresholds, unchanged, to a health set of 1,173 questions built from Wikipedia passages. Wrong answers delivered fall from 57.0% to 9.9%. The target was missed in three languages: with a 10% target, Telugu delivers 12.4%, Bengali 12.2% and Swahili 11.6% wrong answers, while English stays within it at 6.6%. The ranking of risky answers transfers, but the threshold does not, so a real deployment would need health-domain calibration data.

## Cost and toolkits

The guard adds 0.11 seconds per question, from 0.248 to 0.360 seconds on an RTX 5060 Ti, with 8.5 GB peak GPU memory. Methods generating extra answers cost more and detect fewer errors.

The three Track 2 toolkits form one pipeline: CuratorKIT builds the health test set on local models, CircuitKIT finds evidence use peaking at block 23 in all four languages and three reseeded reruns (block 23 writes the probe's layer 24), and AuditKIT produces the paired before-and-after reports.

Tiny Aya is licensed CC-BY-NC-4.0, and the health text is CC BY-SA 4.0.
