import csv
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
HL = ROOT + "data/forecast/"
OUT = ROOT + "paper/figures/"
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA = ("#2a78d6", "#eb6834", "#1baf7a")
GRAY, LIGHT = ("#8a8985", "#c9c8c3")
INK, INK2, GRID = ("#0b0b0b", "#52514e", "#e6e5e1")
RED_T, BLUE_T = ("#d62f2f", "#1f5fd1")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.edgecolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.labelcolor": INK2,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
GAPS = [8, 32, 64, 120]


def rows(path):
    return list(csv.DictReader(open(path)))


def halflife_brier():
    met = rows(HL + "halflife-metrics.csv")
    ref = rows(HL + "halflife-frozen_baseline_reference.csv")

    def m(pop, gap, method):
        r = [
            x
            for x in met
            if x["dataset"] == "confirmation"
            and x["population"] == pop
            and (float(x["horizon_seconds"]) == gap)
            and (x["method"] == method)
        ]
        if not len(r) == 1:
            raise ValueError("Numerical validation failed at source line 54")
        return float(r[0]["brier_take_macro"])

    def b(pop, gap, name):
        r = [
            x
            for x in ref
            if x["dataset"] == "confirmation"
            and x["population"] == pop
            and (float(x["horizon_seconds"]) == gap)
            and (x["baseline"] == name)
        ]
        if not len(r) == 1:
            raise ValueError("Numerical validation failed at source line 58")
        return float(r[0]["brier_take_macro"])

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.35))
    xs = np.arange(4)
    for ax, pop, title in zip(
        axes, ("all", "volatile"), ("All statements", "Volatile statements")
    ):
        series = [
            ("B1 counting", [b(pop, g, "B1") for g in GAPS], GRAY, (0, (3, 2))),
            (
                "Prompted base (same model, no adapter)",
                [m(pop, g, "Base_calibrated") for g in GAPS],
                ORANGE,
                "-",
            ),
            ("B3 contextual ridge", [b(pop, g, "B3") for g in GAPS], AQUA, "-"),
            (
                "Forecaster (adapted)",
                [m(pop, g, "LoRA_calibrated") for g in GAPS],
                BLUE,
                "-",
            ),
        ]
        for i in range(4):
            vals = [s[1][i] for s in series]
            if not vals[0] > vals[1] > vals[2] > vals[3]:
                raise ValueError((pop, GAPS[i], vals))
        ax.axvspan(1.65, 2.35, color=GRID, lw=0, zorder=0)
        ax.text(
            2.0,
            0.02,
            "untrained\ngap",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=6.3,
            color=INK2,
            linespacing=1.0,
        )
        for name, v, c, ls in series:
            ax.plot(
                xs, v, color=c, ls=ls, lw=1.6, marker="o", ms=3.4, label=name, zorder=3
            )
        ax.set_xticks(xs, [f"{g} s" for g in GAPS])
        ax.set_title(title, loc="left")
        ax.set_ylabel("Take-macro Brier (calibrated)" if pop == "all" else "")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(0, None)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(
        h,
        l,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=6.6,
        bbox_to_anchor=(0.5, -0.06),
        handlelength=2.2,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT + "halflife_fig2_brier.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def halflife_recall():
    roc = rows(HL + "halflife-roc-interpolated.csv")

    def r(pop, gap, rate, arm):
        x = [
            y
            for y in roc
            if y["population"] == pop
            and int(y["horizon"]) == gap
            and (abs(float(y["false_rejection_rate"]) - rate) < 1e-09)
            and (y["arm"] == arm)
        ]
        if not len(x) == 1:
            raise ValueError("Numerical validation failed at source line 90")
        x = x[0]
        return (
            float(x["recall"]) * 100,
            float(x["ci95_low"]) * 100,
            float(x["ci95_high"]) * 100,
        )

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.35), sharey=True)
    xs = np.arange(4)
    arms = [
        ("Base", "Prompted base", ORANGE, -0.09),
        ("B3", "B3 contextual ridge", AQUA, 0.0),
        ("LoRA", "Forecaster (adapted)", BLUE, 0.09),
    ]
    for ax, rate in zip(axes, (0.05, 0.1)):
        for key, name, c, dx in arms:
            v = [r("volatile", g, rate, key) for g in GAPS]
            y = [a[0] for a in v]
            lo = [a[0] - a[1] for a in v]
            hi = [a[2] - a[0] for a in v]
            ax.errorbar(
                xs + dx,
                y,
                yerr=[lo, hi],
                color=c,
                lw=1.6,
                marker="o",
                ms=3.4,
                capsize=2,
                elinewidth=0.8,
                label=name,
                zorder=3,
            )
        ax.set_xticks(xs, [f"{g} s" for g in GAPS])
        ax.set_title(f"{int(rate * 100)}% of valid statements rejected", loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Invalid statements caught (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(
        h,
        l,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=6.8,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT + "halflife_fig3_recall.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


ARMS = [
    ("A2", "A2 (generated)", GRAY),
    ("RF", "RF (prefix refit)", ORANGE),
    ("SCENE", "SCENE (with WM future)", BLUE),
    ("C-direct", "C-direct (language model)", AQUA),
]
if __name__ == "__main__":
    halflife_brier()
    halflife_recall()
    print("Aggregate Brier and recall checks passed; figures written to paper/figures.")
