---
title: Knows When It Doesn't Know
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
license: cc-by-nc-4.0
short_description: Tiny Aya health answers that abstain when likely wrong
---

# Knows When It Doesn't Know

Retrieval question answering on Tiny Aya in English, Telugu, Bengali and Swahili. A linear probe on the
layer-24 activation, retrieval similarity and answer log-probability give one risk score, and one global
threshold decides whether the answer is delivered.

This is a research demo, not medical advice. Answers can be wrong even when delivered, and correct answers are
sometimes withheld.

The Space needs an `HF_TOKEN` secret from an account that has accepted the Tiny Aya licence.
For local use, follow the [fresh-checkout instructions](../README.md#run-the-demo-from-a-fresh-checkout).
The thresholds use TyDiQA calibration targets. An experimental checkbox enables one additional passage after abstention.

Licences: Tiny Aya is CC-BY-NC. The passages are from Wikipedia under CC BY-SA 4.0.
