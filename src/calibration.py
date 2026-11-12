"""Per-language abstention thresholds and selective-prediction metrics.

A risk score is higher when the answer is more likely wrong. The model answers
when score <= threshold. Calibration uses conformal risk control on the loss
1[answered and wrong], which is monotone in the threshold. The guarantee covers
the joint rate P(answered and wrong) on data exchangeable with the calibration
split. It does not cover domain shift, which must be measured separately.
"""
import numpy as np
from scipy.stats import beta


def conformal_threshold(scores, wrong, alpha):
    """Largest threshold with (n * empirical_risk + 1) / (n + 1) <= alpha.

    Returns -inf when no threshold qualifies, meaning always abstain.
    """
    scores = np.asarray(scores, dtype=float)
    wrong = np.asarray(wrong, dtype=float)
    n = len(scores)
    if n == 0:
        return -np.inf
    order = np.argsort(scores, kind="stable")
    s, cum_wrong = scores[order], np.cumsum(wrong[order])
    last_of_tie = np.append(s[1:] != s[:-1], True)
    ok = ((cum_wrong + 1) / (n + 1) <= alpha) & last_of_tie
    return float(s[ok].max()) if ok.any() else -np.inf


def per_language_thresholds(scores, wrong, languages, alpha):
    scores, wrong, languages = map(np.asarray, (scores, wrong, languages))
    return {
        lang: conformal_threshold(scores[languages == lang], wrong[languages == lang], alpha)
        for lang in np.unique(languages).tolist()
    }


def selective_threshold(scores, wrong, alpha, delta=0.1, step=0.05):
    """Reproduce the original empirical wrong-among-answered operating point.

    Coverage levels step, 2 * step, ... are tested in order, stopping at the
    first failure. The thresholds come from the same scores used to compute
    the bounds. Use certify_selective_threshold with an independent pilot
    for a fixed-sequence statistical claim.
    """
    scores = np.asarray(scores, dtype=float)
    wrong = np.asarray(wrong, dtype=float)
    order = np.argsort(scores, kind="stable")
    s, cum_wrong = scores[order], np.cumsum(wrong[order])
    chosen = -np.inf
    for level in np.arange(step, 1 + step / 2, step):
        i = min(int(np.ceil(level * len(s))), len(s)) - 1
        while i + 1 < len(s) and s[i + 1] == s[i]:
            i += 1
        k, n = cum_wrong[i], i + 1
        upper = 1.0 if k == n else beta.ppf(1 - delta, k + 1, n - k)
        if upper > alpha:
            break
        chosen = float(s[i])
    return chosen


def coverage_threshold(scores, target):
    """Set a threshold from calibration scores alone, retaining score ties."""
    scores = np.sort(np.asarray(scores, dtype=float))
    if not len(scores) or not 0 < target <= 1 or not np.isfinite(scores).all():
        raise ValueError("Coverage requires finite scores and a target in (0, 1].")
    rank = max(1, int(np.ceil(np.nextafter(target * len(scores), -np.inf))))
    return float(scores[rank - 1])


def certify_selective_threshold(pilot_scores, scores, wrong, alpha=0.2, delta=0.1, step=0.05):
    """Test fixed pilot-derived thresholds on independent labeled examples.

    Conditional on the pilot and for IID certification rows, at each fixed
    threshold the wrong count among accepted rows is binomial conditional on
    the acceptance count. Stop the sequence at the first failure.
    """
    scores = np.asarray(scores, dtype=float)
    wrong = np.asarray(wrong, dtype=bool)
    candidates = sorted({coverage_threshold(pilot_scores, min(level, 1.0))
                         for level in np.arange(step, 1 + step / 2, step)})
    chosen = -np.inf
    for threshold in candidates:
        answered = scores <= threshold
        n = int(answered.sum())
        k = int((answered & wrong).sum())
        upper = beta.ppf(1 - delta, k + 1, n - k) if k < n else 1.0
        if upper > alpha:
            break
        chosen = threshold
    return chosen


def selective_report(scores, wrong, threshold):
    scores = np.asarray(scores, dtype=float)
    wrong = np.asarray(wrong, dtype=bool)
    answered = scores <= threshold
    n_answered = int(answered.sum())
    return {
        "coverage": float(answered.mean()),
        "joint_risk": float((answered & wrong).mean()),
        "selective_risk": float(wrong[answered].mean()) if n_answered else 0.0,
        "n_answered": n_answered,
    }


def auroc(scores, wrong):
    """Probability a wrong answer outscores a right one, ties counted as half."""
    scores = np.asarray(scores, dtype=float)
    wrong = np.asarray(wrong, dtype=bool)
    pos, neg = scores[wrong], scores[~wrong]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    values, inverse = np.unique(np.concatenate([pos, neg]), return_inverse=True)
    counts = np.bincount(inverse, minlength=len(values))
    ranks = np.cumsum(counts) - (counts - 1) / 2.0
    rank_sum = ranks[inverse[: len(pos)]].sum()
    return float((rank_sum - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def aurc(scores, wrong):
    """Area under the risk-coverage curve. Lower is better."""
    scores = np.asarray(scores, dtype=float)
    wrong = np.asarray(wrong, dtype=float)
    order = np.argsort(scores, kind="stable")
    risk = np.cumsum(wrong[order]) / np.arange(1, len(scores) + 1)
    return float(risk.mean())


def bootstrap_ci(metric, *arrays, n_boot=2000, level=0.95, seed=0):
    """Percentile interval for metric(*arrays), resampling rows together."""
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    stats = np.array(
        [metric(*(a[idx] for a in arrays)) for idx in rng.integers(0, n, size=(n_boot, n))]
    )
    tail = (1 - level) / 2 * 100
    return tuple(np.nanpercentile(stats, [tail, 100 - tail]).tolist())


def paired_test(pairs, n_boot=10000, seed=0):
    """Paired bootstrap of candidate minus baseline, with the null centred at zero."""
    diffs = np.array([c - b for b, c in pairs], dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(n_boot)])
    observed = diffs.mean()
    extreme = (np.abs(means - observed) >= abs(observed)).sum()
    low, high = np.percentile(means, [2.5, 97.5])
    return {"n": len(diffs), "delta": float(observed), "ci": [float(low), float(high)],
            "p_value": float((extreme + 1) / (n_boot + 1))}
