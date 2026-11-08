"""The frozen guard as plain arrays: answer activation, similarity and log-probability to risk."""
import numpy as np

LAYER = 24


def _logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


class Guard:
    def __init__(self, path):
        self.w = dict(np.load(path))

    def risk(self, activation, similarity, mean_logprob):
        """Inputs are arrays over examples; activation is (n, hidden). Higher means more likely wrong."""
        w = self.w
        x = (np.asarray(activation, dtype=np.float32) - w["probe_mean"]) / w["probe_scale"]
        probe = _sigmoid(x @ w["probe_coef"] + w["probe_intercept"])
        stack = np.c_[_logit(probe), similarity, mean_logprob]
        return _sigmoid(((stack - w["stack_mean"]) / w["stack_scale"]) @ w["stack_coef"] + w["stack_intercept"])

    def decide(self, risk, rule="joint"):
        return risk <= self.w[f"threshold_{rule}"]
