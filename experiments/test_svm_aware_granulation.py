"""
Proposal 1 experiment: generator x learner comparison for SVM-aware granulation.

Generators: purity-only baseline at 1.0 / 0.95 / 0.8, and SVM-aware (driven by
the same learner it feeds). Learners: GBLSTSVM, ball-weighted linear SVM,
ball-weighted RBF SVM. Datasets: synthetic geometry suite + benchmark CSVs.

Run:  python experiments/test_svm_aware_granulation.py [--seeds 5]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_circles, make_classification
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase8_common import load_dataset_split  # noqa: E402
from src.granular_balls import generate_granular_balls  # noqa: E402
from src.svm_aware_granulation import (  # noqa: E402
    BallSVMLearner,
    GBLSTSVMLearner,
    generate_svm_aware_balls,
)

OUT = Path(__file__).resolve().parent / "output" / "svm_aware_granulation"
REAL = ["breast_cancer", "diabetes", "ecoli2", "vehicle0"]


def synthetic(name: str, seed: int):
    rng = np.random.default_rng(seed)
    n = 600
    if name == "separable":
        X, y = make_blobs(n, centers=[[-2, 0], [2, 0]], cluster_std=0.8, random_state=seed)
    elif name == "overlap":
        X, y = make_blobs(n, centers=[[-1, 0], [1, 0]], cluster_std=1.0, random_state=seed)
    elif name == "elongated":
        X, y = make_blobs(n, centers=[[0, -0.8], [0, 0.8]], cluster_std=0.5, random_state=seed)
        X[:, 0] *= 6
    elif name == "noisy_boundary":
        X, y = make_classification(n, 2, n_informative=2, n_redundant=0, flip_y=0.1,
                                   class_sep=1.0, random_state=seed)
    elif name == "imbalanced":
        X, y = make_classification(n, 5, n_informative=3, weights=[0.9], class_sep=1.0,
                                   random_state=seed)
    elif name == "outliers":
        X, y = make_blobs(n, centers=[[-1.5, 0], [1.5, 0]], cluster_std=0.8, random_state=seed)
        idx = rng.choice(n, 30, replace=False)
        X[idx] += rng.normal(0, 6, (30, 2))
    elif name == "variable_density":
        X1 = rng.normal([-1.5, 0], 0.3, (450, 2))
        X2 = rng.normal([1.5, 0], 1.5, (150, 2))
        X, y = np.vstack([X1, X2]), np.r_[np.zeros(450), np.ones(150)]
    elif name == "circles":
        X, y = make_circles(n, noise=0.08, factor=0.5, random_state=seed)
    elif name == "xor":
        X = rng.uniform(-1, 1, (n, 2))
        y = (X[:, 0] * X[:, 1] > 0).astype(int)
    else:
        raise ValueError(name)
    y = np.where(y == 1, 1.0, -1.0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    return Xtr, Xte, ytr, yte


def real(name: str, seed: int):
    s = load_dataset_split(name, seed)
    Xtr = np.vstack([s.X_train, s.X_val])
    ytr = np.r_[s.y_train, s.y_val].astype(float)
    ytr[ytr == 0] = -1
    yte = s.y_test.astype(float).copy()
    yte[yte == 0] = -1
    return Xtr, s.X_test, ytr, yte


LEARNERS = {
    "GBLSTSVM": lambda: GBLSTSVMLearner(c1=1.0, c2=0.1),
    "BallSVM-lin": lambda: BallSVMLearner(C=1.0, kernel="linear"),
    "BallSVM-rbf": lambda: BallSVMLearner(C=10.0, kernel="rbf"),
}


def generators(X, y, learner):
    for p in (1.0, 0.95, 0.8):
        t = time.perf_counter()
        balls = generate_granular_balls(X, y, purity_threshold=p, min_samples_per_ball=1,
                                        min_ball_count=2)
        yield f"purity{p}", balls, time.perf_counter() - t
    balls, tr = generate_svm_aware_balls(X, y, learner)
    balls = [b for b in balls if b.num_samples >= 2] or balls
    yield "svm_aware", balls, tr.gen_time


def run(seeds: int) -> pd.DataFrame:
    rows = []
    tasks = [(n, "synthetic") for n in ["separable", "overlap", "elongated", "noisy_boundary",
                                        "imbalanced", "outliers", "variable_density",
                                        "circles", "xor"]]
    tasks += [(n, "real") for n in REAL]
    for name, kind in tasks:
        for seed in range(seeds):
            Xtr, Xte, ytr, yte = (synthetic if kind == "synthetic" else real)(name, seed)
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            for lname, make in LEARNERS.items():
                for gname, balls, gtime in generators(Xtr, ytr, make()):
                    np.random.seed(seed)
                    if len({b.label for b in balls}) < 2:
                        continue
                    t = time.perf_counter()
                    dec = make().fit_balls(balls)
                    ftime = time.perf_counter() - t
                    pred = np.where(dec(Xte) > 0, 1.0, -1.0)
                    rows.append(dict(
                        dataset=name, kind=kind, seed=seed, learner=lname, generator=gname,
                        acc=accuracy_score(yte, pred),
                        bal_acc=balanced_accuracy_score(yte, pred),
                        macro_f1=f1_score(yte, pred, average="macro"),
                        n_balls=len(balls),
                        mean_radius=float(np.mean([b.radius for b in balls])),
                        mean_purity=float(np.mean([b.purity for b in balls])),
                        gen_time=gtime, fit_time=ftime,
                    ))
            print(f"done {name} seed {seed}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)
    df = run(args.seeds)
    df.to_csv(OUT / "raw.csv", index=False)
    agg = (df.groupby(["dataset", "learner", "generator"])
             [["bal_acc", "acc", "n_balls", "gen_time"]].mean().round(4))
    agg.to_csv(OUT / "summary.csv")
    overall = (df.groupby(["kind", "learner", "generator"])
                 [["bal_acc", "acc", "macro_f1", "n_balls", "gen_time"]].mean().round(4))
    overall.to_csv(OUT / "overall.csv")
    pd.set_option("display.width", 200)
    print(overall)
    wide = df.groupby(["dataset", "learner", "generator"])["bal_acc"].mean().unstack().round(3)
    print(wide)
    wide.to_csv(OUT / "bal_acc_wide.csv")


if __name__ == "__main__":
    main()
