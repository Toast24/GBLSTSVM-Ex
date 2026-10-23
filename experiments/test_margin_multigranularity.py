"""
Proposal 2 experiment: margin-aware multi-granularity.

A. Pareto sweep (n_balls vs balanced accuracy): purity baseline at
   {0.7, 0.8, 0.9, 0.95, 1.0}, Proposal-1 refiner and Proposal-2 hierarchy at
   theta in {0.4, 0.6, 0.9}, for three learners.
B. Label-noise robustness: 0 / 10 / 20 % training labels flipped.
C. Learner transfer: hierarchy built with pilot learner A, trained with B.

Run:  python experiments/test_margin_multigranularity.py [--seeds 3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.test_svm_aware_granulation import REAL, real, synthetic  # noqa: E402
from src.granular_balls import generate_granular_balls  # noqa: E402
from src.margin_multigranularity import (  # noqa: E402
    WeightedBall,
    WeightedBallSVMLearner,
    WeightedGBLSTSVMLearner,
    build_margin_hierarchy,
)
from src.svm_aware_granulation import generate_svm_aware_balls  # noqa: E402

OUT = Path(__file__).resolve().parent / "output" / "margin_multigranularity"
SYN = ["overlap", "noisy_boundary", "imbalanced", "variable_density", "outliers", "xor"]
LEARNERS = {
    "GBLSTSVM": lambda: WeightedGBLSTSVMLearner(1.0, 0.1),
    "SVM-lin": lambda: WeightedBallSVMLearner(1.0, "linear"),
    "SVM-rbf": lambda: WeightedBallSVMLearner(10.0, "rbf"),
}
THETAS = (0.4, 0.6, 0.9)


def as_weighted(balls):
    return [WeightedBall(b.center, b.radius, b.label, b.purity, b.num_samples, 0) for b in balls]


def configs(X, y, pilot):
    for p in (0.7, 0.8, 0.9, 0.95, 1.0):
        yield "purity", p, as_weighted(generate_granular_balls(X, y, p, 1, 2))
    for t in THETAS:
        b, _ = generate_svm_aware_balls(X, y, pilot, theta=t)
        yield "P1_svm_aware", t, as_weighted([x for x in b if x.num_samples >= 2] or b)
    for t in THETAS:
        yield "P2_margin_multi", t, build_margin_hierarchy(X, y, pilot, theta=t).final


def evaluate(balls, learner, Xte, yte):
    if len({b.label for b in balls}) < 2:
        return np.nan
    pred = np.where(learner.fit_balls(balls)(Xte) > 0, 1.0, -1.0)
    return balanced_accuracy_score(yte, pred)


def load(name, seed, noise=0.0):
    Xtr, Xte, ytr, yte = (real if name in REAL else synthetic)(name, seed)
    if noise:
        rng = np.random.default_rng(1000 + seed)
        idx = rng.choice(len(ytr), int(noise * len(ytr)), replace=False)
        ytr = ytr.copy(); ytr[idx] *= -1
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte), ytr, yte


def pareto_flags(g: pd.DataFrame) -> pd.Series:
    """Point is on the front if no other point has <= balls and >= bal_acc (one strict)."""
    nb, acc = g["n_balls"].values, g["bal_acc"].values
    on = []
    for i in range(len(g)):
        dom = ((nb <= nb[i]) & (acc >= acc[i]) & ((nb < nb[i]) | (acc > acc[i]))).any()
        on.append(not dom)
    return pd.Series(on, index=g.index)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    datasets = SYN + REAL

    # A. Pareto sweep
    rows = []
    for name in datasets:
        for seed in range(args.seeds):
            Xtr, Xte, ytr, yte = load(name, seed)
            for lname, make in LEARNERS.items():
                np.random.seed(seed)
                for method, param, balls in configs(Xtr, ytr, make()):
                    rows.append(dict(dataset=name, seed=seed, learner=lname, method=method,
                                     param=param, n_balls=len(balls),
                                     bal_acc=evaluate(balls, make(), Xte, yte)))
        print("pareto done", name, flush=True)
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "pareto_raw.csv", index=False)
    pts = A.groupby(["dataset", "learner", "method", "param"])[["n_balls", "bal_acc"]].mean().reset_index()
    pts = pts.dropna()
    pts["on_front"] = pts.groupby(["dataset", "learner"], group_keys=False).apply(pareto_flags)
    pts.to_csv(OUT / "pareto_points.csv", index=False)
    front = pts.groupby(["learner", "method"])["on_front"].agg(["sum", "count"])
    front["share"] = (front["sum"] / front["count"]).round(3)
    front.to_csv(OUT / "pareto_front_share.csv")

    # Matched-budget view: best bal_acc each method reaches with <= purity1.0's ball count
    budget = pts[(pts.method == "purity") & (pts.param == 1.0)].set_index(["dataset", "learner"])["n_balls"]
    pts["budget"] = pts.set_index(["dataset", "learner"]).index.map(budget)
    within = pts[pts.n_balls <= pts.budget]
    matched = (within.groupby(["dataset", "learner", "method"])["bal_acc"].max()
                     .unstack().round(3))
    matched.to_csv(OUT / "matched_budget.csv")

    # B. Label noise (default thetas: P1 0.6, P2 0.6; baseline purity 1.0)
    rows = []
    for name in datasets:
        for noise in (0.0, 0.1, 0.2):
            for seed in range(args.seeds):
                Xtr, Xte, ytr, yte = load(name, seed, noise)
                for lname, make in LEARNERS.items():
                    np.random.seed(seed)
                    base = as_weighted(generate_granular_balls(Xtr, ytr, 1.0, 1, 2))
                    p1, _ = generate_svm_aware_balls(Xtr, ytr, make())
                    p1 = as_weighted([x for x in p1 if x.num_samples >= 2] or p1)
                    h = build_margin_hierarchy(Xtr, ytr, make())
                    for m, b in (("purity1.0", base), ("P1_svm_aware", p1), ("P2_margin_multi", h.final)):
                        rows.append(dict(dataset=name, noise=noise, seed=seed, learner=lname,
                                         method=m, n_balls=len(b),
                                         bal_acc=evaluate(b, make(), Xte, yte)))
        print("noise done", name, flush=True)
    B = pd.DataFrame(rows)
    B.to_csv(OUT / "noise_raw.csv", index=False)
    noise_tab = B.groupby(["noise", "learner", "method"])[["bal_acc", "n_balls"]].mean().round(4)
    noise_tab.to_csv(OUT / "noise_summary.csv")

    # C. Learner transfer
    rows = []
    for name in datasets:
        for seed in range(args.seeds):
            Xtr, Xte, ytr, yte = load(name, seed)
            for pilot, mp in LEARNERS.items():
                np.random.seed(seed)
                final = build_margin_hierarchy(Xtr, ytr, mp()).final
                for target, mt in LEARNERS.items():
                    rows.append(dict(dataset=name, seed=seed, pilot=pilot, target=target,
                                     n_balls=len(final), bal_acc=evaluate(final, mt(), Xte, yte)))
    C = pd.DataFrame(rows)
    C.to_csv(OUT / "transfer_raw.csv", index=False)
    transfer = C.groupby(["pilot", "target"])["bal_acc"].mean().unstack().round(4)
    transfer.to_csv(OUT / "transfer_matrix.csv")

    pd.set_option("display.width", 200)
    print("\n== Pareto-front share (fraction of a method's configs that are non-dominated) ==")
    print(front)
    print("\n== Mean best bal_acc within purity-1.0 ball budget ==")
    print(matched.groupby(level="learner").mean().round(4))
    print("\n== Label noise ==")
    print(noise_tab["bal_acc"].unstack())
    print("\n== Learner transfer (rows: pilot used to build hierarchy, cols: learner trained) ==")
    print(transfer)


if __name__ == "__main__":
    main()
