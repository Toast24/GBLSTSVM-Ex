"""Generate Phase 8 plots from experiment outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "output" / "phase8"


def load_json(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def plot_reliability_vs_purity() -> None:
    data = load_json("granulation_sensitivity.json")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (ds, ds_data) in zip(axes, data["datasets"].items()):
        purities = []
        rel_means = []
        frac_ge99 = []
        for entry in ds_data["results"]:
            p = entry["purity_threshold"]
            cr_model = entry.get("models", {}).get("CR-GBLSTSVM", {})
            cr_stats = cr_model.get("ball_stats", {})
            if not cr_stats:
                continue
            purities.append(p)
            rel_means.append(cr_stats.get("reliability_mean") or 0)
            frac_ge99.append(cr_stats.get("frac_reliability_ge_0_99") or 0)
        ax.plot(purities, rel_means, "o-", label="mean reliability")
        ax.plot(purities, frac_ge99, "s--", label="frac >= 0.99")
        ax.set_title(ds)
        ax.set_xlabel("purity_threshold")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("value")
    fig.suptitle("Reliability vs purity threshold (CR-GBLSTSVM)")
    fig.tight_layout()
    fig.savefig(OUT / "reliability_vs_purity.png", dpi=150)
    plt.close(fig)


def plot_macro_f1_comparison() -> None:
    fp = load_json("final_protocol.json")
    datasets = list(fp["datasets"].keys())
    models = [
        "Original/reference GBLSTSVM",
        "Clean GBLSTSVM",
        "CW-GBLSTSVM",
        "R-GBLSTSVM",
        "CR-GBLSTSVM",
    ]
    x = np.arange(len(datasets))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        vals = []
        for ds in datasets:
            m = fp["datasets"][ds]["models"].get(model, {})
            vals.append(m.get("test", {}).get("macro_f1", 0) or 0)
        ax.bar(x + i * width, vals, width, label=model.split()[0])
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Test Macro F1")
    ax.set_title("Final protocol: test Macro F1 by model")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "macro_f1_comparison.png", dpi=150)
    plt.close(fig)


def plot_cr_vs_cw_delta() -> None:
    ms = load_json("multiseed.json")
    datasets = list(ms["datasets"].keys())
    cr_cw = [ms["datasets"][d]["delta_CR_minus_CW_macro_f1_mean"] for d in datasets]
    cr_cw_std = [ms["datasets"][d]["delta_CR_minus_CW_macro_f1_std"] for d in datasets]
    cw_clean = [ms["datasets"][d]["delta_CW_minus_clean_macro_f1_mean"] for d in datasets]
    cw_clean_std = [ms["datasets"][d]["delta_CW_minus_clean_macro_f1_std"] for d in datasets]
    x = np.arange(len(datasets))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, cw_clean, width, yerr=cw_clean_std, label="CW - Clean", capsize=4)
    ax.bar(x + width / 2, cr_cw, width, yerr=cr_cw_std, label="CR - CW", capsize=4)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Delta Macro F1 (test)")
    ax.set_title("Multi-seed stability of deltas")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cr_vs_cw_delta.png", dpi=150)
    plt.close(fig)


def plot_scaling_comparison() -> None:
    data = load_json("scaling_ablation.json")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (ds, ds_data) in zip(axes, data["datasets"].items()):
        scalings = []
        clean_f1 = []
        cw_f1 = []
        cr_f1 = []
        for entry in ds_data["results"]:
            scalings.append(entry["scaling"])
            clean_f1.append(entry["models"]["Clean GBLSTSVM"]["test"]["macro_f1"])
            cw_f1.append(entry["models"]["CW-GBLSTSVM"]["test"]["macro_f1"])
            cr_f1.append(entry["models"]["CR-GBLSTSVM"]["test"]["macro_f1"])
        x = np.arange(len(scalings))
        w = 0.25
        ax.bar(x - w, clean_f1, w, label="Clean")
        ax.bar(x, cw_f1, w, label="CW")
        ax.bar(x + w, cr_f1, w, label="CR")
        ax.set_xticks(x)
        ax.set_xticklabels(scalings)
        ax.set_title(ds)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Test Macro F1")
    fig.suptitle("Scaling ablation (test Macro F1)")
    fig.tight_layout()
    fig.savefig(OUT / "scaling_comparison.png", dpi=150)
    plt.close(fig)


def plot_seed_stability() -> None:
    import pandas as pd

    csv_path = OUT / "multiseed.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, ds in zip(axes, df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        ax.plot(sub["seed"], sub["Clean GBLSTSVM_test_macro_f1"], "o-", label="Clean")
        ax.plot(sub["seed"], sub["CW-GBLSTSVM_test_macro_f1"], "s-", label="CW")
        ax.plot(sub["seed"], sub["CR-GBLSTSVM_test_macro_f1"], "^-", label="CR")
        ax.set_title(ds)
        ax.set_xlabel("seed")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Test Macro F1")
    fig.suptitle("Multi-seed test Macro F1 stability")
    fig.tight_layout()
    fig.savefig(OUT / "seed_stability.png", dpi=150)
    plt.close(fig)


def plot_reliability_distribution() -> None:
    mech = load_json("mechanism_diagnostics.json")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3))
    for ax, (ds, entry) in zip(axes, mech["datasets"].items()):
        rel = entry["reliability"]
        labels = ["<0.90", "<0.95", ">=0.95", ">=0.99"]
        vals = [
            rel.get("frac_reliability_lt_0_90", 0),
            rel.get("frac_reliability_lt_0_95", 0),
            rel.get("frac_reliability_ge_0_95", 0),
            rel.get("frac_reliability_ge_0_99", 0),
        ]
        ax.bar(labels, vals)
        ax.set_title(ds)
        ax.set_ylabel("fraction of balls")
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Reliability distribution (final protocol, CR model)")
    fig.tight_layout()
    fig.savefig(OUT / "reliability_distribution.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_reliability_vs_purity()
    plot_macro_f1_comparison()
    plot_cr_vs_cw_delta()
    plot_scaling_comparison()
    plot_seed_stability()
    plot_reliability_distribution()
    print(f"Plots written to {OUT}")


if __name__ == "__main__":
    main()
