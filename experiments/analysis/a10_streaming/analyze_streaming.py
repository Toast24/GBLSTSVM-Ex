"""
A10 tables and figures. Unit = dataset (mean over seeds); Wilcoxon + percentile
bootstrap (seed 42, 10k) + Holm over all contrasts x noise levels.

  python experiments/analysis/a10_streaming/analyze_streaming.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401  (log-axis labels without mathtext)
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments" / "output" / "a10_streaming"
RNG = np.random.default_rng(42)
ARMS = ["RAW_REFIT", "CLASS_MOM", "BALL_MOM", "BALL_MOM_W", "PUB_STREAM", "PUB_REGRAN"]
CONTRASTS = [("BALL_MOM_W", "CLASS_MOM"), ("CLASS_MOM", "PUB_STREAM"), ("CLASS_MOM", "PUB_REGRAN"),
             ("PUB_REGRAN", "PUB_STREAM")]
COLORS = {"RAW_REFIT": "#1B2430", "CLASS_MOM": "#1F6F8B", "BALL_MOM": "#7FB3C8", "BALL_MOM_W": "#5C9E74",
          "PUB_STREAM": "#E07A3F", "PUB_REGRAN": "#B84A4A"}


def holm(p):
    order, m, run = np.argsort(p), len(p), 0.0
    adj = np.empty(m)
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def main() -> int:
    df = pd.read_csv(OUT / "runs.csv")
    unlearn = df[df.arm == "UNLEARN_CHECK"]
    df = df[df.arm != "UNLEARN_CHECK"]
    last = df[df.batch == df.batch.max()]

    summ: dict = {
        "exact_class_mom_max_relerr": float(df[df.arm == "CLASS_MOM"].relerr_vs_refit.max()),
        "exact_ball_mom_max_relerr": float(df[df.arm == "BALL_MOM"].relerr_vs_refit.max()),
        "unlearn_max_relerr": float(unlearn.relerr_vs_refit.max()),
        "n_streams": int(last.groupby(["dataset", "seed", "noise"]).ngroups),
    }

    f1 = last.pivot_table(index=["noise", "dataset"], columns="arm", values="test_f1", aggfunc="mean")
    f1.groupby("noise").mean()[ARMS].round(4).to_csv(OUT / "table_final_f1.csv")
    summ["final_f1"] = {str(k): v for k, v in f1.groupby("noise").mean()[ARMS].round(4).to_dict("index").items()}

    rows = []
    for noise, g in f1.groupby("noise"):
        for a, b in CONTRASTS:
            d = (g[a] - g[b]).dropna().to_numpy()
            boot = RNG.choice(d, size=(10_000, len(d))).mean(1)
            rows.append({"noise": noise, "contrast": f"{a} - {b}", "mean_delta": d.mean(),
                         "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5),
                         "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                         "p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0})
    ct = pd.DataFrame(rows)
    ct["p_holm"] = holm(ct.p.to_numpy())
    ct.round(4).to_csv(OUT / "table_contrasts.csv", index=False)
    summ["contrasts"] = ct.round(4).to_dict("records")

    cost = last.groupby("arm")[["update_seconds", "stored_floats"]].median()
    big = last[last.n_seen >= 2000].groupby("arm")[["update_seconds", "stored_floats"]].median()
    cost.round(6).to_csv(OUT / "table_cost_all.csv")
    big.round(6).to_csv(OUT / "table_cost_spambase.csv")
    summ["cost_median_last_batch"] = cost.round(6).to_dict("index")
    summ["cost_large_n"] = big.round(6).to_dict("index")

    # Figures: F1 and cost along the stream (dataset means)
    curve = df.groupby(["noise", "arm", "batch"]).test_f1.mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    plt.rcParams.update({"font.family": "Calibri"})
    for ax, noise in zip(axes, sorted(df.noise.unique())):
        for arm in ["CLASS_MOM", "BALL_MOM_W", "PUB_STREAM", "PUB_REGRAN"]:
            c = curve[(curve.noise == noise) & (curve.arm == arm)]
            ax.plot(c.batch, c.test_f1, marker="o", ms=4, lw=2, color=COLORS[arm],
                    label="CLASS_MOM = RAW_REFIT" if arm == "CLASS_MOM" else arm)
        ax.set_title(f"label noise {int(noise * 100)}%")
        ax.set_xlabel("batches seen")
        ax.grid(color="#E3E9EE", lw=.6); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("test macro-F1 (mean of 17 datasets)")
    axes[0].legend(frameon=False, fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "f1_along_stream.png", dpi=200); plt.close(fig)

    sp = df[(df.n_seen > 0) & (df.dataset == "spambase") & (df.noise == 0.0)]
    sp = sp.groupby(["arm", "batch"])[["update_seconds", "stored_floats"]].median().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for arm in ARMS:
        c = sp[sp.arm == arm]
        axes[0].plot(c.batch, c.update_seconds * 1000, marker="o", ms=4, lw=2, color=COLORS[arm], label=arm)
        axes[1].plot(c.batch, c.stored_floats, marker="o", ms=4, lw=2, color=COLORS[arm], label=arm)
    axes[0].set_ylabel("update time (ms)"); axes[1].set_ylabel("numbers stored")
    for ax in axes:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.4g}"))
        ax.set_xlabel("batches seen (spambase, n = 3,220)")
        ax.grid(color="#E3E9EE", lw=.6); ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "cost_along_stream.png", dpi=200); plt.close(fig)

    (OUT / "summary.json").write_text(json.dumps(summ, indent=2, default=float))
    print(json.dumps(summ, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
