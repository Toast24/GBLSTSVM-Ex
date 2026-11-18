"""Figures for the midsem deck. Run with the GBenv interpreter from the repo root."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

TEAL, ORANGE, INK, MUTED = "#1F6F8B", "#E07A3F", "#1B2430", "#7A8B99"
plt.rcParams.update({"font.family": "Calibri", "font.size": 13, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})


def granulate(X, y, rho, rng):
    """Recursive KMeans(k=2) purity splitting, as in the baseline generator."""
    from sklearn.cluster import KMeans
    out, stack = [], [np.arange(len(X))]
    while stack:
        idx = stack.pop()
        p = max(np.mean(y[idx] == 1), np.mean(y[idx] == -1))
        if p >= rho or len(idx) < 2:
            out.append(idx)
            continue
        lab = KMeans(2, n_init=3, random_state=int(rng.integers(1e6))).fit_predict(X[idx])
        if lab.min() == lab.max():
            out.append(idx)
            continue
        stack += [idx[lab == 0], idx[lab == 1]]
    return out


def fig_granulation():
    rng = np.random.default_rng(3)
    A = rng.normal([0, 0], [1.0, 0.7], (140, 2))
    B = rng.normal([2.0, 1.2], [0.8, 0.9], (45, 2))
    X, y = np.vstack([A, B]), np.r_[-np.ones(140), np.ones(45)]
    balls = granulate(X, y, 0.95, rng)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, title in zip(axes, ["Raw training samples", "Granular balls (ρ = 0.95)"]):
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, color=INK)
        for s in ax.spines.values(): s.set_visible(False)
    axes[0].scatter(*A.T, s=14, c=TEAL, alpha=.7, label="majority (−1)")
    axes[0].scatter(*B.T, s=14, c=ORANGE, alpha=.9, label="minority (+1)")
    axes[0].legend(frameon=False, loc="lower right")
    axes[1].scatter(*X.T, s=5, c=[TEAL if v < 0 else ORANGE for v in y], alpha=.25)
    for idx in balls:
        c = X[idx].mean(0); r = np.linalg.norm(X[idx] - c, axis=1).mean()
        lab = 1 if y[idx].mean() > 0 else -1; col = ORANGE if lab > 0 else TEAL
        if len(idx) < 2:
            axes[1].scatter(*c, marker="x", s=40, c="#B84A4A", zorder=3)
        else:
            axes[1].add_patch(plt.Circle(c, max(r, .05), fill=False, ec=col, lw=1.6))
    axes[1].scatter([], [], marker="x", c="#B84A4A", label="singleton (deleted by del_ball)")
    axes[1].legend(frameon=False, loc="lower right")
    for ax in axes: ax.set_xlim(X[:, 0].min() - .5, X[:, 0].max() + .5); ax.set_ylim(X[:, 1].min() - .5, X[:, 1].max() + .5)
    fig.tight_layout(); fig.savefig(OUT / "granulation.png", dpi=200); plt.close(fig)


def fig_theta_eta():
    df = pd.read_csv(ROOT / "experiments/output/a8_angle_gate/sweep/runs.csv")
    df = df[(df.path == "PARTITION") & (df.arm == "L1")].copy()
    df["theta"] = (df.theta1_vs_raw + df.theta2_vs_raw) / 2
    df["eta"] = (df.within_frac_pos_partition + df.within_frac_neg_partition) / 2
    g = df.groupby(["dataset", "rho"])[["theta", "eta"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    sc = ax.scatter(g.eta, g.theta, c=g.rho, cmap="viridis", s=38, edgecolor="white", lw=.5)
    cb = fig.colorbar(sc, ax=ax); cb.set_label("granulation threshold ρ")
    ax.set_xlabel("η = share of class variance discarded by center-only balls")
    ax.set_ylabel("θ(L1, RAW) in degrees")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#E3E9EE", lw=.6); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(OUT / "theta_vs_eta.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    fig_granulation()
    fig_theta_eta()
    print("ok", sorted(p.name for p in OUT.iterdir()))
