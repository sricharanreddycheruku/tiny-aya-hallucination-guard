"""Article figures, drawn from the final reports."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GUARD, ALWAYS = "#2a78d6", "#eb6834"
LANGUAGES = {"en": "English", "te": "Telugu", "bn": "Bengali", "sw": "Swahili"}
METHODS = {"guard": "Guard", "probe": "Probe alone", "p_true": "P(True)", "similarity": "Retrieval similarity",
           "logprob": "Log-probability", "agreement": "Self-consistency", "consensus": "Cross-lingual consensus"}
OUT = Path("../runs/figures")
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.size": 11,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def load(name):
    return json.load(open(f"../runs/final/{name}.json"))


def bars(ax, left, right, labels, names):
    width = 0.34
    for offset, values, colour, name in ((-0.18, left, ALWAYS, names[0]), (0.18, right, GUARD, names[1])):
        marks = ax.bar([i + offset for i in range(len(values))], [v * 100 for v in values], width, color=colour, label=name)
        ax.bar_label(marks, fmt="%.1f", padding=2, fontsize=9, color=MUTED)
    ax.set_xticks(range(len(labels)), labels)
    ax.tick_params(length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def outcomes(report):
    rows = report["languages"]
    always = [rows[k]["wrong_if_always_answering"] for k in LANGUAGES]
    guard = [rows[k]["guard"]["joint"]["joint_risk"] for k in LANGUAGES]
    right_guard = [rows[k]["guard"]["joint"]["coverage"] - rows[k]["guard"]["joint"]["joint_risk"] for k in LANGUAGES]
    return always, guard, [1 - v for v in always], right_guard


def before_after():
    always, guard, right_always, right_guard = outcomes(load("rag_test"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    names = ("Always answer", "With guard")
    bars(axes[0], always, guard, list(LANGUAGES.values()), names)
    bars(axes[1], right_always, right_guard, list(LANGUAGES.values()), names)
    axes[0].set_title("Wrong answers delivered")
    axes[1].set_title("Right answers delivered")
    axes[0].set_ylabel("% of all questions")
    axes[0].set_ylim(0, 70)
    axes[0].legend(frameon=False, loc="upper right", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "before_after.png", dpi=180)


def separation():
    rows = load("rag_test")["languages"]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.6), sharex=True, sharey=True)
    names = list(METHODS)[::-1]
    for ax, (code, language) in zip(axes, LANGUAGES.items()):
        for y, name in enumerate(names):
            score = rows[code]["scores"][name]
            colour = GUARD if name == "guard" else MUTED
            low, high = score["ci"]
            ax.plot([low, high], [y, y], color=colour, linewidth=2, solid_capstyle="round")
            ax.plot(score["auroc"], y, "o", color=colour, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.set_title(f"{language} (n={rows[code]['n']})")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    axes[0].set_yticks(range(len(names)), [METHODS[n] for n in names])
    axes[0].set_xlim(0.5, 1.0)
    fig.supxlabel("AUROC for detecting a wrong answer, with 95% interval", color=MUTED, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "separation.png", dpi=180)


def domain_shift():
    test = load("rag_test")["languages"]
    health = load("health_strict")["languages"]
    left = [test[k]["guard"]["joint"]["joint_risk"] for k in LANGUAGES]
    right = [health[k]["guard"]["joint"]["joint_risk"] for k in LANGUAGES]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    width = 0.34
    for offset, values, colour, name in ((-0.18, left, GUARD, "TyDiQA test"), (0.18, right, ALWAYS, "Health, strict set")):
        marks = ax.bar([i + offset for i in range(4)], [v * 100 for v in values], width, color=colour, label=name)
        ax.bar_label(marks, fmt="%.1f", padding=2, fontsize=9, color=MUTED)
    ax.axhline(10, color=INK, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(-0.45, 10.3, "10% target", ha="left", fontsize=9, color=MUTED)
    ax.set_xticks(range(4), list(LANGUAGES.values()))
    ax.set_ylabel("Wrong and delivered, %")
    ax.set_title("Same frozen threshold, new domain")
    ax.tick_params(length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    ax.set_ylim(0, max(right + left) * 100 * 1.35)
    fig.tight_layout()
    fig.savefig(OUT / "domain_shift.png", dpi=180)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    before_after()
    separation()
    if Path("../runs/final/health_strict.json").exists():
        domain_shift()
