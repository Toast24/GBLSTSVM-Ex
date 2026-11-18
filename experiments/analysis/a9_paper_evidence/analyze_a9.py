"""
A9 tables and figures. Unit of inference = dataset (mean over seeds); Wilcoxon
signed-rank + percentile bootstrap (seed 42, 10k) + Holm within each table.

  python experiments/analysis/a9_paper_evidence/analyze_a9.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments" / "output" / "a9_paper_evidence"
FIG = OUT / "figures"
RNG = np.random.default_rng(42)

CONTRASTS = [
    ("P_L1", "P_L0", "population weights"),
    ("P_DIAG", "P_L1", "diagonal scatter"),
    ("P_FULL", "P_DIAG", "off-diagonal scatter"),
    ("P_DIAG", "P_ISO", "where mass goes"),
    ("P_ISO", "P_L1", "isotropic mass"),
    ("D_FULL", "D_L1", "scatter on deletion path"),
    ("P_L1", "D_L1", "cost of deletion (L1)"),
    ("RAW", "D_FULL", "cost of deletion (exact)"),
    ("RAW", "PUB", "RAW vs published"),
]


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def boot_ci(x: np.ndarray) -> tuple[float, float]:
    b = RNG.choice(x, size=(10_000, len(x)), replace=True).mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def contrast_rows(df: pd.DataFrame, col: str) -> list[dict]:
    ds = df.pivot_table(index="dataset", columns="arm", values=col, aggfunc="mean")
    rows = []
    for a, b, what in CONTRASTS:
        if a not in ds or b not in ds:
            continue
        d = (ds[a] - ds[b]).dropna().to_numpy()
        p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        rows.append({"contrast": f"{a} - {b}", "meaning": what, "mean_delta": d.mean(), "p": p,
                     "wins": int((d > 0).sum()), "losses": int((d < 0).sum())})
    for r, h in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = h
    return rows


def stability() -> pd.DataFrame:
    """Same contrasts across independent train/val split draws (code rule)."""
    sources = {
        "draw0": OUT / "heldout_ladder" / "runs.csv",
        "draw1": OUT / "heldout_ladder_draw1" / "runs.csv",
        "draw2": OUT / "heldout_ladder_draw2" / "runs.csv",
        "salted_a9": OUT / "prefix_hash_salted" / "heldout_ladder" / "runs.csv",
        "salted_a8e2": ROOT / "experiments/output/a8_angle_gate/linear_heldout/runs.csv",
        "std_draw0": OUT / "heldout_ladder_std" / "runs.csv",
        "std_draw1": OUT / "heldout_ladder_std_draw1" / "runs.csv",
        "std_draw2": OUT / "heldout_ladder_std_draw2" / "runs.csv",
        "std_salted_a9": OUT / "prefix_hash_salted" / "heldout_ladder_std" / "runs.csv",
    }
    recs = []
    for name, path in sources.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df.ok == True]  # noqa: E712
        for r in contrast_rows(df, "test_f1_repo"):
            recs.append({"source": name, "scaling": "std" if name.startswith("std") else "unscaled", **r})
    t = pd.DataFrame(recs)
    t.to_csv(OUT / "table_split_stability.csv", index=False)
    return t


def ladder_tables(tag: str) -> dict:
    df = pd.read_csv(OUT / tag / "runs.csv")
    df = df[df.ok == True]  # noqa: E712
    res = {}
    for rule in ("repo", "paper"):
        col = f"test_f1_{rule}"
        ds = df.pivot_table(index="dataset", columns="arm", values=col, aggfunc="mean")
        arm_means = ds.mean().sort_values(ascending=False)
        rows = []
        for a, b, what in CONTRASTS:
            d = (ds[a] - ds[b]).dropna().to_numpy()
            p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            lo, hi = boot_ci(d)
            rows.append({"contrast": f"{a} - {b}", "meaning": what, "mean_delta": d.mean(),
                         "ci_lo": lo, "ci_hi": hi, "wins": int((d > 0).sum()),
                         "losses": int((d < 0).sum()), "n": len(d), "p": p})
        for r, h in zip(rows, holm([r["p"] for r in rows])):
            r["p_holm"] = h
        ct = pd.DataFrame(rows)
        ct.to_csv(OUT / f"table_{tag}_{rule}_contrasts.csv", index=False)
        arm_means.rename("mean_f1").to_csv(OUT / f"table_{tag}_{rule}_arms.csv")
        res[rule] = {"arms": arm_means.round(4).to_dict(), "contrasts": ct.round(4).to_dict("records")}
    return res


def deletion_table() -> pd.DataFrame:
    df = pd.read_csv(OUT / "heldout_ladder" / "runs.csv")
    df = df[df.ok == True]  # noqa: E712
    f1 = df.pivot_table(index="dataset", columns="arm", values="test_f1_repo", aggfunc="mean")
    drop = df.groupby("dataset")[["dropped_frac", "dropped_frac_pos", "dropped_frac_neg", "n_train"]].mean()
    t = drop.join(pd.DataFrame({"raw_minus_dfull": f1.RAW - f1.D_FULL,
                                "pl1_minus_dl1": f1.P_L1 - f1.D_L1}))
    t = t.sort_values("dropped_frac", ascending=False)
    t.to_csv(OUT / "table_deletion_cost.csv")
    return t


def closed_form(tag: str) -> dict:
    df = pd.read_csv(OUT / tag / "runs.csv")
    out = {"identity_resid_max": float(df.identity_resid.max()), "bound_ok_frac": float(df.bound_ok.mean()),
           "n_rows": len(df)}
    cells = df.groupby(["dataset", "rho", "arm"])[["theta", "q", "eta", "relerr"]].mean().reset_index()
    cells["logq"] = np.log10(cells.q)
    per_arm = {}
    for arm, g in cells.groupby("arm"):
        within_q = [spearmanr(h.logq, h.theta).statistic for _, h in g.groupby("dataset") if len(h) > 2]
        within_e = [spearmanr(h.eta, h.theta).statistic for _, h in g.groupby("dataset") if len(h) > 2]
        per_arm[arm] = {
            "spearman_theta_q": float(spearmanr(g.logq, g.theta).statistic),
            "spearman_theta_eta": float(spearmanr(g.eta, g.theta).statistic),
            "within_q_positive": int(np.sum(np.array(within_q) > 0)),
            "within_eta_positive": int(np.sum(np.array(within_e) > 0)),
            "n_datasets": len(within_q),
            "median_within_q": float(np.nanmedian(within_q)),
            "median_within_eta": float(np.nanmedian(within_e)),
            "median_relerr_over_q": float(np.median(g.relerr / g.q)),
        }
    # Is theta(arm) ordering predicted by q ordering within a cell?
    piv_t = cells.pivot_table(index=["dataset", "rho"], columns="arm", values="theta")
    piv_q = cells.pivot_table(index=["dataset", "rho"], columns="arm", values="q")
    agree = [(np.argsort(piv_t.loc[i].values) == np.argsort(piv_q.loc[i].values)).all() for i in piv_t.index]
    out["per_arm"] = per_arm
    out["cell_order_agreement"] = float(np.mean(agree))
    cells.to_csv(OUT / f"table_{tag}_cells.csv", index=False)
    return out


def figures() -> None:
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "Calibri", "font.size": 13})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, tag, title in zip(axes, ("closed_form_std", "closed_form"), ("standardized", "unscaled")):
        c = pd.read_csv(OUT / f"table_{tag}_cells.csv")
        c = c[c.arm == "L1"]
        sc = ax.scatter(c.eta, c.theta, c=c.rho, cmap="viridis", s=30, edgecolor="white", lw=.4)
        ax.set_title(f"θ(L1, RAW) vs η, {title}")
        ax.set_xlabel("η = share of class variance discarded"); ax.set_ylabel("θ in degrees")
        ax.spines[["top", "right"]].set_visible(False); ax.grid(color="#E3E9EE", lw=.6); ax.set_axisbelow(True)
    fig.colorbar(sc, ax=axes, label="ρ")
    fig.savefig(FIG / "theta_vs_eta_both.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    c = pd.read_csv(OUT / "table_closed_form_std_cells.csv")
    for ax, arm in zip(axes, ("L1", "ISO", "DIAG")):
        g = c[c.arm == arm]
        ax.scatter(g.logq, g.theta, c=g.rho, cmap="viridis", s=26, edgecolor="white", lw=.4)
        r = spearmanr(g.logq, g.theta).statistic
        ax.set_title(f"{arm}: Spearman {r:.2f}")
        ax.set_xlabel("log₁₀ ‖M⁻¹ΔM‖₂  (computed before solving)")
        ax.spines[["top", "right"]].set_visible(False); ax.grid(color="#E3E9EE", lw=.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("θ to RAW (degrees)")
    fig.tight_layout(); fig.savefig(FIG / "theta_vs_q_std.png", dpi=200); plt.close(fig)


def main() -> int:
    summary = {
        "ladder_unscaled": ladder_tables("heldout_ladder"),
        "ladder_std": ladder_tables("heldout_ladder_std"),
        "closed_form_unscaled": closed_form("closed_form"),
        "closed_form_std": closed_form("closed_form_std"),
    }
    d = deletion_table()
    summary["deletion"] = {"mean_dropped": float(d.dropped_frac.mean()),
                           "mean_dropped_pos": float(d.dropped_frac_pos.mean()),
                           "mean_dropped_neg": float(d.dropped_frac_neg.mean()),
                           "mean_raw_minus_dfull": float(d.raw_minus_dfull.mean()),
                           "spearman_drop_vs_cost": float(spearmanr(d.dropped_frac, d.raw_minus_dfull).statistic)}
    st = stability()
    g = st.groupby(["scaling", "contrast"])
    summary["split_stability"] = g.agg(
        draws=("mean_delta", "size"), delta_min=("mean_delta", "min"), delta_max=("mean_delta", "max"),
        sig_draws=("p_holm", lambda p: int((p < 0.05).sum())),
    ).reset_index().round(4).to_dict("records")
    figures()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
