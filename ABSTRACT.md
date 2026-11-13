# Knows When It Doesn't Know

**Lexsi Track 2b, Problem 13 | Hallucination detection and abstention**

Tiny Aya can produce a plausible answer even when its retrieved passage provides little evidence. We built an activation-based guard for English, Telugu, Bengali, and Swahili. The guard decides whether to deliver an answer after Tiny Aya Fire reads a retrieved passage.

## Method

A linear probe uses the layer-24 hidden state at the final answer token to estimate error risk. We evaluate the probe alone and a stacker that adds retrieval similarity and mean answer log-probability. We fit on a probe split and set thresholds on a separate calibration split. Correctness uses automatic matching against reference answers. CuratorKIT builds the multilingual health benchmark, CircuitKIT examines the activations involved in evidence use, and AuditKIT produces paired before-and-after reports.

## Results

In a follow-up run on 1,721 TyDiQA test questions, the activation probe answered **55.1%** and delivered **748 correct and 200 wrong answers**. Retrieval-augmented generation without abstention delivered 881 correct and 840 wrong answers, so the probe reduced wrong deliveries by **76.2%** while answering most questions. At similar coverage, the comparator using retrieval similarity and log-probability without activations delivered 689 correct and 301 wrong answers. The probe gained 59 correct deliveries and avoided 101 wrong ones. Bengali coverage remained below half.

The original locked benchmark tested a more conservative guard. Wrong answers delivered fell from 48.6% to 6.6% of all questions, a 42.07 percentage-point drop with a 95% confidence interval of 39.80 to 44.39 points. Correct answers delivered fell from 51.4% to 39.7%, and coverage was 46.3%. In the follow-up replay, the majority-coverage probe delivered 69 more correct answers and 90 more wrong answers than the published guard.

## Domain shift and cost

On 1,173 health questions, the original TyDiQA-calibrated guard reduced overall wrong delivery from 57.0% to 9.9%. Telugu, Bengali, and Swahili still exceeded the 10% target individually, motivating health-specific recalibration. The original guard added 0.11 seconds per question on an RTX 5060 Ti and used 8.5 GB of peak GPU memory.

The repository includes the code, benchmark reports, and reproducibility instructions. Tiny Aya is released under CC-BY-NC-4.0.
