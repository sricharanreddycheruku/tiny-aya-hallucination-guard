"""Export the frozen guard to arrays and check it reproduces the fitted pipeline."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze import column
from calibration import conformal_threshold, selective_threshold
from combine import LAYER_INDEX, load_splits
from final import fit_guard, risk
from guard import Guard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-run", default="../runs/rag_probe")
    parser.add_argument("--calibration-run", default="../runs/rag_calibration")
    parser.add_argument("--out", default="../runs/guard/guard_fire.npz")
    args = parser.parse_args()

    rows, train, train_acts, train_keep, cal, cal_acts, keep = load_splits(args.train_run, args.calibration_run)
    contexts = {r["id"]: r["context"] for r in rows}
    probe, stacker = fit_guard(train, train_acts, train_keep, contexts, args.train_run, LAYER_INDEX)
    expected, _, similarity = risk(probe, stacker, cal, cal_acts, keep, LAYER_INDEX, contexts, f"{args.calibration_run}/similarity.json")
    wrong = column(cal, "wrong")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        probe_mean=probe[0].mean_.astype(np.float32), probe_scale=probe[0].scale_.astype(np.float32),
        probe_coef=probe[1].coef_[0].astype(np.float32), probe_intercept=probe[1].intercept_[0],
        stack_mean=stacker[0].mean_, stack_scale=stacker[0].scale_,
        stack_coef=stacker[1].coef_[0], stack_intercept=stacker[1].intercept_[0],
        threshold_joint=conformal_threshold(expected, wrong, 0.10),
        threshold_selective=selective_threshold(expected, wrong, 0.20),
    )
    guard = Guard(args.out)
    x = cal_acts["answer"][keep, LAYER_INDEX].astype(np.float32)
    got = guard.risk(x, similarity, column(cal, "mean_logprob"))
    print("max difference from fitted pipeline", float(np.abs(got - expected).max()))
    print("agreement on joint decision", float((guard.decide(got) == (expected <= guard.w["threshold_joint"])).mean()))
    print(json.dumps({k: float(guard.w[k]) for k in ("threshold_joint", "threshold_selective")}))


if __name__ == "__main__":
    main()
