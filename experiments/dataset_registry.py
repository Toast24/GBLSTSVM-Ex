"""
Discover and load benchmark datasets for screening experiments.

Supports headerless CSVs in Data/Data/ and UCI downloads in datasets/uci/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from experiments.phase8_common import (
    DATA_DIR,
    DatasetSplit,
    inspect_dataset,
    map_to_binary,
    normalize_labels,
    stratified_three_way_split,
)

ROOT = Path(__file__).resolve().parents[1]
UCI_DIR = ROOT / "datasets" / "uci"

ONEHOT_DEFAULT = True

# Paper names that are not available in the repository (do not block screening).
UNAVAILABLE_DATASET_ALIASES = {
    "breast-cancer",
    "breast-cancer-wisc",
    "conn-bench-sonar-mines-rocks",
    "diabetes-upload",
    "electrical",
    "fourclass",
    "liver-disorders",
    "twonorm",
}


def discover_available_datasets() -> list[dict]:
    """Return loadable dataset descriptors sorted by name."""
    found: list[dict] = []

    for path in sorted(DATA_DIR.glob("*.csv")):
        found.append(
            {
                "name": path.stem,
                "path": path,
                "source": "data_dir",
                "key": f"data:{path.name}",
            }
        )

    if UCI_DIR.is_dir():
        for sub in sorted(UCI_DIR.iterdir()):
            if not sub.is_dir():
                continue
            data_path = sub / "data.csv"
            if data_path.exists():
                found.append(
                    {
                        "name": sub.name,
                        "path": data_path,
                        "source": "uci",
                        "key": f"uci:{sub.name}",
                    }
                )

    return found


def _encode_features(df: pd.DataFrame, *, onehot: bool = ONEHOT_DEFAULT) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for col in df.columns:
        series = df[col]
        if series.dtype == object or not pd.api.types.is_numeric_dtype(series.dtype):
            cleaned = series.astype(str).str.strip()
            cleaned = cleaned.replace({"": np.nan, "?": np.nan, "nan": np.nan, "NaN": np.nan})
            if onehot:
                parts.append(pd.get_dummies(cleaned, prefix=col, dtype=float))
            else:
                codes, _ = pd.factorize(cleaned, sort=True)
                coded = pd.Series(codes.astype(float), index=df.index, name=col)
                coded.loc[cleaned.isna()] = np.nan
                parts.append(coded.to_frame())
        else:
            parts.append(pd.to_numeric(series, errors="coerce").to_frame(name=col))
    return pd.concat(parts, axis=1)


def _impute_missing(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if not np.isnan(X).any():
        return X
    out = X.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        mask = np.isnan(col)
        if not mask.any():
            continue
        fill = float(np.nanmedian(col)) if np.isfinite(np.nanmedian(col)) else 0.0
        out[mask, j] = fill
    return out


def _reduce_to_binary(y_raw: pd.Series) -> tuple[np.ndarray, dict]:
    y = y_raw.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    counts = y.value_counts().to_dict()
    info = {
        "n_classes": len(counts),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "label_values": sorted(counts.keys(), key=lambda v: (-counts[v], str(v))),
    }

    if info["n_classes"] < 2:
        raise ValueError(f"Expected at least 2 classes; got {info['n_classes']}.")

    if info["n_classes"] > 2:
        keep = sorted(counts, key=counts.get, reverse=True)[:2]
        mask = y.isin(keep)
        if mask.sum() < 20:
            raise ValueError("Too few samples after reducing to two largest classes.")
        y = y[mask]
        info["binary_reduction"] = f"kept two largest classes: {keep}"
        info["n_samples_after_reduction"] = int(mask.sum())
    else:
        info["binary_reduction"] = None

    # Map to numeric raw labels for map_to_binary
    label_values = sorted(y.unique().tolist(), key=lambda v: (str(type(v)), str(v)))
    counts_bin = y.value_counts().to_dict()
    info.update(
        {
            "n_classes": 2,
            "class_counts": {str(k): int(v) for k, v in counts_bin.items()},
            "label_values": label_values,
            "imbalance_ratio": float(max(counts_bin.values()) / min(counts_bin.values())),
        }
    )

    mapping = {raw: float(i) for i, raw in enumerate(label_values)}
    y_num = y.map(mapping).to_numpy(dtype=float)
    return y_num, info


def load_uci_dataset(
    path: Path,
    *,
    onehot: bool = ONEHOT_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, dict]:
    df = pd.read_csv(path)
    target_cols = [c for c in df.columns if c.startswith("target_")]
    if len(target_cols) != 1:
        raise ValueError(f"Expected exactly one target_* column; got {target_cols}.")

    target_col = target_cols[0]
    feature_df = df.drop(columns=target_cols)
    feature_df = _encode_features(feature_df, onehot=onehot)
    y_num, label_info = _reduce_to_binary(df[target_col])

    if label_info.get("binary_reduction"):
        keep_labels = set(label_info["label_values"])
        raw_y = df[target_col].astype(str).str.strip()
        row_mask = raw_y.isin(keep_labels).to_numpy()
        feature_df = feature_df.loc[row_mask].reset_index(drop=True)

    X = _impute_missing(feature_df.to_numpy(dtype=float))

    info = {
        "name": path.parent.name,
        "path": str(path),
        "shape": [int(X.shape[0]), int(X.shape[1] + 1)],
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "missing_values": int(np.isnan(feature_df.to_numpy(dtype=float)).sum()),
        "duplicate_rows": int(pd.DataFrame(np.column_stack([X, y_num])).duplicated().sum()),
        "duplicates_retained": True,
        "source": "uci",
        "target_column": target_col,
        "onehot": onehot,
        **label_info,
    }
    return X, y_num, info


def load_dataset_split_from_descriptor(
    descriptor: dict,
    seed: int,
    *,
    onehot: bool = ONEHOT_DEFAULT,
) -> DatasetSplit:
    path: Path = descriptor["path"]
    source: Literal["data_dir", "uci"] = descriptor["source"]

    if source == "data_dir":
        return load_dataset_split_by_name(path.name, seed=seed)

    X, y_raw, info = load_uci_dataset(path, onehot=onehot)
    y, label_mapping = map_to_binary(y_raw, info)
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_three_way_split(X, y, seed)
    y_train_n = normalize_labels(y_train)
    n_pos = int(np.sum(y_train_n == 1.0))
    n_neg = int(np.sum(y_train_n == -1.0))
    minority_label = 1.0 if n_pos <= n_neg else -1.0
    return DatasetSplit(
        name=info["name"],
        info=info,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        minority_label=minority_label,
        label_mapping=label_mapping,
    )


def load_dataset_split_by_name(name: str, seed: int) -> DatasetSplit:
    from experiments.phase8_common import load_dataset_split

    if not name.endswith(".csv"):
        name = f"{name}.csv"
    return load_dataset_split(name, seed=seed)


def try_load_descriptor(
    descriptor: dict,
    seed: int,
    *,
    onehot: bool = ONEHOT_DEFAULT,
) -> tuple[DatasetSplit | None, str | None]:
    try:
        return load_dataset_split_from_descriptor(descriptor, seed=seed, onehot=onehot), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"
