import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from calibration import (aurc, auroc, bootstrap_ci, certify_selective_threshold,
                         conformal_threshold, coverage_threshold,
                         per_language_thresholds, selective_report,
                         paired_test, selective_threshold)


def synthetic(n, error_rate, separation, rng):
    wrong = rng.random(n) < error_rate
    return rng.normal(0, 1, n) + separation * wrong, wrong


def test_auroc_known_values():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auroc([0.8, 0.9, 0.1, 0.2], [0, 0, 1, 1]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]) == 0.5


def test_aurc_prefers_better_ranking():
    wrong = [0, 0, 1, 1]
    assert aurc([0.1, 0.2, 0.8, 0.9], wrong) < aurc([0.9, 0.8, 0.2, 0.1], wrong)


def test_no_valid_threshold_means_always_abstain():
    assert conformal_threshold([0.1, 0.2], [1, 1], 0.05) == -np.inf
    assert conformal_threshold([], [], 0.1) == -np.inf
    assert selective_report([0.1, 0.2], [1, 1], -np.inf)["coverage"] == 0.0


def test_ties_are_not_split():
    t = conformal_threshold([0.5] * 10 + [0.9] * 10, [0] * 9 + [1] + [1] * 10, 0.2)
    assert t == 0.5


def test_risk_controlled_on_average_across_trials():
    rng = np.random.default_rng(1)
    alpha, risks = 0.1, []
    for _ in range(400):
        s, w = synthetic(400, 0.5, 2.0, rng)
        t = conformal_threshold(s, w, alpha)
        s2, w2 = synthetic(2000, 0.5, 2.0, rng)
        risks.append(selective_report(s2, w2, t)["joint_risk"])
    assert np.mean(risks) <= alpha


def test_per_language_beats_global_on_the_weak_language():
    rng = np.random.default_rng(2)
    alpha = 0.1
    s_en, w_en = synthetic(4000, 0.2, 2.0, rng)
    s_te, w_te = synthetic(4000, 0.7, 1.0, rng)
    scores = np.concatenate([s_en, s_te])
    wrong = np.concatenate([w_en, w_te])
    langs = np.array(["en"] * 4000 + ["te"] * 4000)
    pooled = conformal_threshold(scores, wrong, alpha)
    own = per_language_thresholds(scores, wrong, langs, alpha)
    s2, w2 = synthetic(20000, 0.7, 1.0, rng)
    assert selective_report(s2, w2, pooled)["joint_risk"] > alpha
    assert selective_report(s2, w2, own["te"])["joint_risk"] <= alpha + 0.01


def test_bootstrap_interval_contains_estimate():
    rng = np.random.default_rng(3)
    s, w = synthetic(500, 0.4, 1.5, rng)
    low, high = bootstrap_ci(auroc, s, w, n_boot=300)
    assert low < auroc(s, w) < high


def test_selective_threshold_controls_wrong_among_answered():
    rng = np.random.default_rng(3)
    scores = rng.uniform(size=4000)
    wrong = rng.uniform(size=4000) < scores
    threshold = selective_threshold(scores, wrong, 0.2)
    assert 0.2 < threshold < 0.4
    fresh = rng.uniform(size=200000)
    answered = fresh <= threshold
    assert (rng.uniform(size=200000) < fresh)[answered].mean() <= 0.2


def test_selective_threshold_abstains_when_nothing_qualifies():
    assert selective_threshold(np.arange(100.0), np.ones(100), 0.1) == -np.inf


def test_selective_threshold_survives_a_noisy_start():
    wrong = np.array([1] * 6 + [0] * 94 + [0] * 1500 + [1] * 400)
    scores = np.arange(len(wrong), dtype=float)
    t = selective_threshold(scores, wrong, 0.2)
    assert selective_report(scores, wrong, t)["coverage"] >= 0.75


def test_paired_test_separates_real_effect_from_noise():
    rng = np.random.default_rng(4)
    base = rng.random(500)
    assert paired_test(list(zip(base, base - 0.4)), n_boot=500)["p_value"] < 0.01
    noise = rng.normal(0, 1, 500)
    noise -= noise.mean()
    assert paired_test(list(zip(np.zeros(500), noise)), n_boot=500)["p_value"] > 0.5


def test_coverage_threshold_keeps_ties():
    scores = [0.1, 0.2, 0.2, 0.9]
    assert coverage_threshold(scores, 0.5) == 0.2
    assert coverage_threshold(scores, 0.6) == 0.2


def test_certification_uses_pilot_candidates_and_stops_on_failure():
    pilot = [0.1, 0.2, 0.3, 0.4]
    scores = np.repeat([0.09, 0.19, 0.29, 0.39], 100)
    wrong = np.array([False] * 200 + [True] * 100 + [False] * 100)
    assert certify_selective_threshold(pilot, scores, wrong, step=0.25) == 0.2
    assert certify_selective_threshold(pilot, [0.3], [False], step=0.25) == -np.inf
