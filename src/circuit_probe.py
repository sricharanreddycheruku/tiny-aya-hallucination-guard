"""Probes on circuit-selected layers against the fixed layer 24 probe, on the calibration split.

Layers are ranked by the EAP-IG importance of the attention heads CircuitKIT kept,
so the choice uses no calibration labels. MLP scores are left out because pruning
covers heads only and MLP 0 alone outweighs every other node. Block b writes
hidden state b + 1.

HallucinationDetector.train_probes is not used for training: ProbeTrainer reads only
the first layer of each activation dict, and every epoch repeats a forward pass per
example. The same LinearProbe and ProbeTrainer run here on cached activations, one
layer at a time, and the detector stores the result. ProbeTrainer keeps its best
state by reference, so the best epoch is copied here.
"""
import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from circuitkit.applications.common_utils.hallucination_detection import HallucinationDetector
from circuitkit.applications.common_utils.linear_probe import LinearProbe, ProbeTrainer
from circuitkit.artifacts import eap_to_artifact
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from analyze import column
from calibration import aurc, auroc, bootstrap_ci
from combine import LANGS, load_splits

BASE_LAYER = 24


def head_mass(scores_path, pruned_path):
    stored = json.load(open(scores_path))
    pruned = set(torch.load(pruned_path, weights_only=False))
    kept = {k: v for k, v in stored["node_scores"].items() if k not in pruned}
    mass = defaultdict(float)
    for name, value in kept.items():
        if name.startswith("A"):
            mass[int(name[1:].split(".")[0])] += abs(value)
    return dict(mass), kept, stored["model"]


def fit_linear(x, y, validation, epochs, seed):
    """Train one CircuitKIT probe and fold the scaling into its weights, so it reads raw activations."""
    torch.manual_seed(seed)
    mean, std = x.mean(0), x.std(0) + 1e-6
    scaled = torch.tensor((x - mean) / std)
    labels = torch.tensor(y, dtype=torch.float32)
    train = DataLoader(TensorDataset(scaled[~validation], labels[~validation]), batch_size=256, shuffle=True)
    held = DataLoader(TensorDataset(scaled[validation], labels[validation]), batch_size=1024)
    probe = LinearProbe(x.shape[1])
    trainer = ProbeTrainer(probe, device="cpu", learning_rate=1e-3, weight_decay=1e-2)
    best_auroc, best_state = -1.0, None
    for _ in range(epochs):
        value = trainer.train_epoch(train, held)[3]
        if value > best_auroc:
            best_auroc, best_state = value, copy.deepcopy(probe.state_dict())
    probe.load_state_dict(best_state)
    with torch.no_grad():
        probe.linear.weight /= torch.tensor(std)
        probe.linear.bias -= (probe.linear.weight * torch.tensor(mean)).sum()
    return probe.eval(), best_auroc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-run", default="../runs/all_probe")
    parser.add_argument("--calibration-run", default="../runs/all_calibration")
    parser.add_argument("--scores", default="../runs/circuit/evidence_scores.json")
    parser.add_argument("--pruned", default="../runs/circuit/evidence.pt")
    parser.add_argument("--top", type=int, nargs="+", default=[1, 4, 12])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--out", default="../runs/circuit/probe_comparison.json")
    parser.add_argument("--probes", default="../runs/circuit/detector_probes.pt")
    args = parser.parse_args()

    _, train, train_acts, train_keep, cal, cal_acts, keep = load_splits(args.train_run, args.calibration_run)
    layers = train_acts["layers"].tolist()
    x_train_all = train_acts["answer"][train_keep]
    x_cal_all = cal_acts["answer"][keep]
    y_train, y_cal = column(train, "wrong"), column(cal, "wrong")
    languages = column(cal, "language")
    validation = np.array([int(hashlib.sha1(r["title"].encode()).hexdigest(), 16) % 7 == 0 for r in train])
    print("rows", len(train), "validation", int(validation.sum()), "calibration", len(cal), flush=True)

    mass, kept, model_id = head_mass(args.scores, args.pruned)
    ranked = [b for b in sorted(mass, key=mass.get, reverse=True) if b + 1 in layers]
    print("blocks by kept head importance", [(b, round(mass[b], 3)) for b in ranked[:12]], flush=True)

    def features(block_output, x):
        return x[:, layers.index(block_output)].astype(np.float32)

    probes, val_auroc, cal_scores = {}, {}, {}
    for block in sorted(set(ranked[: max(args.top)]) | {BASE_LAYER - 1}):
        x = features(block + 1, x_train_all)
        probes[block], val_auroc[block] = fit_linear(x, y_train, validation, args.epochs, seed=block)
        with torch.no_grad():
            cal_scores[block] = probes[block](torch.tensor(features(block + 1, x_cal_all))).squeeze(-1).numpy()
        print("trained block", block, "validation auroc", round(val_auroc[block], 3), flush=True)

    fixed = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=500))
    fixed.fit(features(BASE_LAYER, x_train_all), y_train)
    base = fixed.predict_proba(features(BASE_LAYER, x_cal_all))[:, 1]
    scores = {"layer24_fixed": base, "layer24_circuitkit": cal_scores[BASE_LAYER - 1]}
    for k in args.top:
        scores[f"circuit_top{k}"] = np.mean([cal_scores[b] for b in ranked[:k]], axis=0)

    def gain(a, b, wrong):
        return auroc(a, wrong) - auroc(b, wrong)

    report = {"ranked_blocks": ranked, "head_mass": {str(b): mass[b] for b in ranked}, "languages": {}}
    for language in LANGS:
        m = languages == language
        entry = {}
        for name, s in scores.items():
            entry[name] = {
                "auroc": auroc(s[m], y_cal[m]), "aurc": aurc(s[m], y_cal[m]),
                "gain_over_fixed_ci": bootstrap_ci(gain, s[m], base[m], y_cal[m], n_boot=args.n_boot),
            }
            print(language, name, round(entry[name]["auroc"], 3), "aurc", round(entry[name]["aurc"], 3),
                  "gain ci", [round(v, 3) for v in entry[name]["gain_over_fixed_ci"]], flush=True)
        report["languages"][language] = entry

    curve = {}
    for layer in layers:
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=300))
        s = probe.fit(features(layer, x_train_all), y_train).predict_proba(features(layer, x_cal_all))[:, 1]
        curve[layer] = {l: auroc(s[languages == l], y_cal[languages == l]) for l in LANGS}
        print("layer", layer, {k: round(v, 3) for k, v in curve[layer].items()}, flush=True)
    report["layer_curve"] = curve
    shared = [l for l in layers if l - 1 in mass]
    report["importance_vs_auroc_spearman"] = float(spearmanr(
        [mass[l - 1] for l in shared], [np.mean(list(curve[l].values())) for l in shared]
    ).statistic)
    print("spearman head importance vs mean auroc", round(report["importance_vs_auroc_spearman"], 3))

    artifact = eap_to_artifact(node_scores=kept, model_id=model_id, task="evidence", dataset="tydiqa_secondary")
    detector = HallucinationDetector(SimpleNamespace(config=None), artifact, {}, device="cpu")
    detector.probes = {b: probes[b] for b in ranked[: max(args.top)]}
    detector.save_probes(args.probes)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
