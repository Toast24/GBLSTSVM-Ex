"""
A8 analysis: dataset-level statistics for the gate, rho sweep, and held-out ladder.

  python experiments/analysis/a8_angle_gate/analyze_a8.py
Writes experiments/output/a8_angle_gate/summary_*.csv and prints markdown-ready tables.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments" / "output" / "a8_angle_gate"
RNG_SEED = 42


def boot_ci(d: np.ndarray, n: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(RNG_SEED)
    m = d[rng.integers(0, d.size, (n, d.size))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(da: pd.Series, db: pd.Series) -> dict:
    d = (da - db).dropna().to_numpy()
    p = stats.wilcoxon(d).pvalue if d.size > 1 and np.any(d != 0) else float("nan")
    lo, hi = boot_ci(d)
    return {"n": d.size, "mean": d.mean(), "median": np.median(d), "wins_a_lower": int((d < 0).sum()),
            "p_wilcoxon": p, "ci_lo": lo, "ci_hi": hi}


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    m = len(p)
    adj, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[k]))
        adj[k] = run
    return pd.Series(adj)[p.index]


def gate_tables(tag: str) -> None:
    path = OUT / tag / "runs.csv"
    if not path.exists():
        return
    r = pd.read_csv(path)
    r["theta"] = (r.theta1_vs_raw + r.theta2_vs_raw) / 2
    print(f"\n## {tag}: partition-path geometry at c = C_RAW_HO (dataset means)\n")
    p = r[(r.c_source == "C_RAW_HO") & (r.path.isin(["PARTITION", "RAW"]))]
    ds = p.pivot_table(index="dataset", columns="arm", values="theta", aggfunc="mean")
    print(ds.describe().loc[["mean", "50%"]].round(2).to_string())
    arms = ["L0", "L1", "DIAG", "ISO"]
    rows = []
    for a, b in combinations(arms, 2):
        rows.append({"contrast": f"{a}-{b}", **paired(ds[a], ds[b])})
    t = pd.DataFrame(rows).set_index("contrast")
    t["p_holm"] = holm(t.p_wilcoxon)
    print("\n" + t.round(4).to_string())
    t.to_csv(OUT / f"summary_{tag}_contrasts.csv")

    # Decomposition of A7's theta_FULL (unscaled only has C_A6_SHARED)
    dec = r[(r.arm.isin(["FULL"]))].groupby(["c_source", "path"])[["theta", "theta1_vs_retained"]].mean()
    print("\nFULL decomposition (theta vs RAW / vs RAW_RETAINED):\n" + dec.round(3).to_string())

    # Mechanism: center-only loss vs discarded within-ball variance fraction
    g = p.drop_duplicates(["dataset", "seed"]).set_index(["dataset", "seed"])
    wf = ((g.within_frac_pos_partition + g.within_frac_neg_partition) / 2).groupby("dataset").mean()
    mis = ((g.iso_mismatch_pos_partition + g.iso_mismatch_neg_partition) / 2).groupby("dataset").mean()
    mech = []
    for name, x, y in (
        ("theta_L1 vs within_frac", wf, ds["L1"]),
        ("theta_DIAG vs within_frac", wf, ds["DIAG"]),
        ("theta_ISO vs iso_mismatch", mis, ds["ISO"]),
        ("theta_ISO-theta_L1 vs iso_mismatch", mis, ds["ISO"] - ds["L1"]),
        ("theta_ISO-theta_L1 vs within_frac", wf, ds["ISO"] - ds["L1"]),
    ):
        rho, pv = stats.spearmanr(x.loc[y.index], y)
        mech.append({"relation": name, "spearman": rho, "p": pv})
    # Run-level (dataset x seed) version for within-frac
    rl = p[p.arm == "L1"].copy()
    rl["wf"] = (rl.within_frac_pos_partition + rl.within_frac_neg_partition) / 2
    rho, pv = stats.spearmanr(rl.wf, rl.theta)
    mech.append({"relation": "theta_L1 vs within_frac (run level)", "spearman": rho, "p": pv})
    m = pd.DataFrame(mech)
    print("\nMechanism correlations (dataset level unless noted):\n" + m.round(4).to_string(index=False))
    m.to_csv(OUT / f"summary_{tag}_mechanism.csv", index=False)
    per_ds = ds.join(wf.rename("within_frac")).join(mis.rename("iso_mismatch"))
    drop = g.dropped_frac.groupby("dataset").mean().rename("dropped_frac")
    per_ds.join(drop).round(3).to_csv(OUT / f"summary_{tag}_per_dataset.csv")


def sweep_tables() -> None:
    path = OUT / "sweep" / "runs.csv"
    if not path.exists():
        return
    r = pd.read_csv(path)
    r["theta"] = (r.theta1_vs_raw + r.theta2_vs_raw) / 2
    r["wf"] = (r.within_frac_pos_partition + r.within_frac_neg_partition) / 2
    p = r[r.path == "PARTITION"]
    print("\n## rho sweep (partition path, c = C_RAW_HO): dataset-mean theta by rho\n")
    t = p.groupby(["rho", "arm", "dataset"]).theta.mean().groupby(["rho", "arm"]).mean().unstack()
    t = t.join(p.drop_duplicates(["dataset", "seed", "rho"]).groupby("rho").agg(
        within_frac=("wf", "mean"), n_balls=("n_balls_partition", "mean")))
    print(t.round(2).to_string())
    t.to_csv(OUT / "summary_sweep.csv")
    f1 = r.groupby(["rho", "path", "arm"]).test_f1_repo.mean().unstack(["path", "arm"])
    f1.round(4).to_csv(OUT / "summary_sweep_f1.csv")
    # Pooled dose-response across rho levels (dataset x rho cells)
    cell = p[p.arm == "L1"].groupby(["dataset", "rho"])[["theta", "wf"]].mean()
    rho_s, pv = stats.spearmanr(cell.wf, cell.theta)
    within = cell.groupby(level=0).apply(lambda c: stats.spearmanr(c.wf, c.theta)[0] if len(c) > 2 else np.nan)
    print(f"\ntheta_L1 vs within_frac across (dataset, rho) cells: spearman={rho_s:.3f}, p={pv:.2e}")
    print(f"within-dataset spearman (rho sweep): median={within.median():.3f}, "
          f"positive in {int((within > 0).sum())}/{within.notna().sum()} datasets")


def heldout_tables() -> None:
    path = OUT / "linear_heldout" / "runs.csv"
    if not path.exists():
        return
    r = pd.read_csv(path)
    r = r[r.ok == True]  # noqa: E712
    print(f"\n## held-out-tuned linear ladder ({r.dataset.nunique()} datasets)\n")
    for metric in ("test_f1_repo", "test_f1_paper"):
        ds = r.pivot_table(index="dataset", columns="arm", values=metric, aggfunc="mean")
        print(f"\n{metric}: mean over datasets\n" + ds.mean().round(4).to_string())
        rows = [{"contrast": f"{a}-RAW", **paired(ds[a], ds["RAW"])} for a in ds.columns if a != "RAW"]
        rows += [{"contrast": "P_L1-D_L1", **paired(ds["P_L1"], ds["D_L1"])},
                 {"contrast": "D_FULL-D_L1", **paired(ds["D_FULL"], ds["D_L1"])},
                 {"contrast": "P_L1-P_L0", **paired(ds["P_L1"], ds["P_L0"])}]
        t = pd.DataFrame(rows).set_index("contrast")
        t["p_holm"] = holm(t.p_wilcoxon)
        print(t.round(4).to_string())
        t.to_csv(OUT / f"summary_heldout_{metric}.csv")
    cost = r.groupby("arm")[["representation_seconds", "solve_seconds"]].mean()
    print("\nmean seconds per full-train fit:\n" + cost.round(5).to_string())
    coll = r.pivot_table(index="dataset", columns="arm", values="collapsed_repo", aggfunc="mean").mean()
    print("\ncollapse rate (RULE_REPO):\n" + coll.round(3).to_string())


def kernel_tables() -> None:
    path = OUT / "kernel_pilot" / "runs.csv"
    if not path.exists():
        return
    r = pd.read_csv(path)
    print(f"\n## kernel pilot ({r.dataset.nunique()} datasets, {r.seed.nunique()} seeds)\n")
    cols = ["test_f1", "test_f1_at_raw_hp", "agree_with_raw", "score_spearman_raw", "rep_seconds", "fit_seconds"]
    per = r.groupby(["arm", "dataset"])[cols].mean().groupby("arm").mean()
    per["k_over_n"] = (r.k / r.n_train).groupby(r.arm).mean()
    print(per.round(4).to_string())
    per.to_csv(OUT / "summary_kernel_arms.csv")
    ds = r.pivot_table(index="dataset", columns="arm", values="test_f1", aggfunc="mean")
    ag = r.pivot_table(index="dataset", columns="arm", values="agree_with_raw", aggfunc="mean")
    pairs = [("K_GB_EXACT", "K_RAW"), ("K_GB_L1", "K_RAW"), ("K_GB_L0", "K_RAW"), ("K_GB_MEAN", "K_RAW"),
             ("K_GB_EXACT", "K_GB_L1"), ("K_GB_EXACT", "K_GB_MEAN"), ("K_GB_MEAN", "K_GB_L1"),
             ("K_GB_L1", "K_GB_L0"), ("K_GB_EXACT", "K_KM_EXACT"), ("K_GB_EXACT", "K_RND_EXACT")]
    for name, tab in (("test_f1", ds), ("agree_with_raw", ag)):
        rows = [{"contrast": f"{a}-{b}", **paired(tab[a], tab[b])} for a, b in pairs if b in tab]
        t = pd.DataFrame(rows).set_index("contrast")
        t["p_holm"] = holm(t.p_wilcoxon)
        print(f"\n{name}:\n" + t.round(4).to_string())
        t.to_csv(OUT / f"summary_kernel_{name}.csv")
    big = r[r.n_train >= 500].groupby("arm")[["rep_seconds", "fit_seconds"]].mean()
    big["total"] = big.sum(1)
    print("\nseconds per full-train fit, datasets with n_train >= 500:\n" + big.round(4).to_string())


def noise_tables() -> None:
    path = OUT / "noise_sweep" / "runs.csv"
    if not path.exists():
        return
    r = pd.read_csv(path)
    r = r[r.ok == True]  # noqa: E712
    print(f"\n## label-noise sweep ({r.dataset.nunique()} datasets)\n")
    for metric in ("test_f1_repo", "test_f1_paper"):
        t = r.groupby(["noise", "arm", "dataset"])[metric].mean().groupby(["noise", "arm"]).mean().unstack()
        print(f"\n{metric} (dataset mean):\n" + t.round(4).to_string())
        t.to_csv(OUT / f"summary_noise_{metric}.csv")
    contrasts = [("D_FULL", "RAW"), ("ENN_RAW", "RAW"), ("D_FULL", "ENN_RAW"), ("D_FULL", "RDROP_RAW"),
                 ("PUB", "RAW"), ("D_FULL", "D_L1"), ("P_L1", "RAW")]
    out = []
    for metric in ("test_f1_repo", "test_f1_paper"):
        for p, sub in r.groupby("noise"):
            ds = sub.pivot_table(index="dataset", columns="arm", values=metric, aggfunc="mean")
            rows = [{"metric": metric, "noise": p, "contrast": f"{a}-{b}", **paired(ds[a], ds[b])}
                    for a, b in contrasts]
            t = pd.DataFrame(rows)
            t["p_holm"] = holm(t.set_index("contrast").p_wilcoxon).to_numpy()
            out.append(t)
    ct = pd.concat(out)
    ct.to_csv(OUT / "summary_noise_contrasts.csv", index=False)
    show = ct[["metric", "noise", "contrast", "mean", "wins_a_lower", "p_wilcoxon", "p_holm"]]
    print("\nPaired contrasts (Holm within metric x noise level):\n" + show.round(4).to_string(index=False))
    q = r.drop_duplicates(["dataset", "seed", "noise"])
    fq = q.groupby(["noise", "dataset"])[[c for c in q.columns if c.startswith(("gb_", "enn_"))]].mean().groupby("noise").mean()
    print("\nFilter quality (precision vs base rate = noise level):\n" + fq.round(3).to_string())
    fq.to_csv(OUT / "summary_noise_filter_quality.csv")


def main() -> int:
    noise_tables()
    for tag in ("gate", "gate_std"):
        gate_tables(tag)
    sweep_tables()
    heldout_tables()
    kernel_tables()
    return 0


if __name__ == "__main__":
    sys.exit(main())
