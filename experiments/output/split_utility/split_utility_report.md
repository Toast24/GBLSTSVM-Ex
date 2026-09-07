# Split-utility granulation counterfactual

Read-only replay of frozen recursive KMeans(k=2) granulation with split-utility gates (rho=0.95, min_samples_per_ball=1, del_ball num_data=2), followed by **unchanged Clean GBLSTSVM** (no CW / CA / Topo).

Runs: **510** result rows across 51 dataset×seed pairs.

## Research questions

### 1. Does split utility reduce minority loss?

Split-utility gates raise mean minority survival on 9/9 non-B0 configs vs B0 (B0=0.583). Yes — refusing high-separation splits reduces del_ball minority loss.

### 2. Does it reduce deleted singleton children?

Singleton-ball deletions (n=1 leaves removed by del_ball) decrease on 9/9 configs (B0 mean deleted singleton balls=179.88). Yes — high-separation rejections primarily block isolated singleton peels.

### 3. Does it improve minority F1?

Minority F1 improves vs B0 on 2/9 configs (mean Δ=-0.0411). Weak or inconsistent — not enough evidence for a classifier-level win.

### 4. Does it hurt majority performance?

Majority survival drops on 0/9 configs (B0 majority survival=0.750). Majority survival is largely unchanged on average.

### 5. Which threshold is best?

Best pre-specified config: **SU3** (p75), mean minority F1=0.441, Δ vs B0=+0.0239, mean Δ macro F1=-0.0443. SU3 p75 is the strongest minority-F1 gate, but macro F1 still trails B0 on average.

Ranking rule: Among non-B0 configs, rank by mean delta minority_f1 vs B0, then delta minority_survival, then mean macro_f1. No classifier hyperparameter tuning.

## Baseline (B0) representation + classifier means

- balls: 88.8
- minority survival: 0.583
- majority survival: 0.750
- minority small-ball rate: 0.749
- deleted balls: 179.9
- deleted singleton balls: 179.9
- mean purity: 0.997
- mean radius: 187.9909
- ball size p50: 16.9
- heterogeneous balls: 1.0
- accuracy: 0.669
- macro F1: 0.548
- minority F1: 0.433
- G-mean: 0.395

## All configurations (means vs B0)

| mode | threshold | balls | min surv | maj surv | min small | del balls | del sing | rej splits | macro F1 | min F1 | gmean | acc | Δ min surv | Δ min F1 | Δ macro F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | baseline | 88.8 | 0.583 | 0.750 | 0.749 | 179.9 | 179.9 | 0.0 | 0.548 | 0.433 | 0.395 | 0.669 | +0.0000 | +0.0000 | +0.0000 |
| SU1 | 1.0 | 15.2 | 1.000 | 1.000 | 0.080 | 0.0 | 0.0 | 9.9 | 0.547 | 0.457 | 0.387 | 0.655 | +0.4171 | +0.0127 | -0.0142 |
| SU1 | 1.5 | 39.7 | 0.999 | 0.998 | 0.251 | 1.0 | 1.0 | 24.4 | 0.522 | 0.363 | 0.317 | 0.673 | +0.4160 | -0.0532 | -0.0389 |
| SU1 | 2.0 | 67.5 | 0.979 | 0.986 | 0.337 | 10.3 | 10.3 | 36.4 | 0.522 | 0.397 | 0.354 | 0.645 | +0.3958 | -0.0313 | -0.0407 |
| SU1 | 2.5 | 98.5 | 0.933 | 0.963 | 0.491 | 33.8 | 33.8 | 47.4 | 0.510 | 0.343 | 0.329 | 0.655 | +0.3497 | -0.0639 | -0.0530 |
| SU1 | 3.0 | 113.3 | 0.897 | 0.933 | 0.591 | 50.0 | 50.0 | 49.5 | 0.493 | 0.349 | 0.295 | 0.640 | +0.3145 | -0.0716 | -0.0574 |
| SU2 | median_parent_le5 | 127.3 | 0.943 | 0.961 | 0.747 | 26.4 | 26.4 | 65.8 | 0.474 | 0.332 | 0.273 | 0.623 | +0.3605 | -0.1012 | -0.0738 |
| SU3 | p25 | 106.1 | 0.878 | 0.908 | 0.566 | 59.9 | 59.9 | 44.3 | 0.470 | 0.341 | 0.257 | 0.618 | +0.2947 | -0.0665 | -0.0651 |
| SU3 | p50 | 73.8 | 0.948 | 0.960 | 0.335 | 27.4 | 27.4 | 37.6 | 0.505 | 0.384 | 0.350 | 0.632 | +0.3650 | -0.0191 | -0.0418 |
| SU3 | p75 | 41.9 | 0.989 | 0.990 | 0.153 | 8.0 | 8.0 | 26.1 | 0.543 | 0.441 | 0.444 | 0.648 | +0.4058 | +0.0239 | -0.0141 |

## Hypothesis verdict

Not confirmed at classifier level: minority survival improves strongly, but mean macro F1 falls vs B0; only SU3 p75 shows a small mean minority F1 gain (+0.024).

## Per dataset × seed × mode (classifier columns)

| dataset | seed | mode | threshold | n_balls | min surv | maj surv | macro F1 | min F1 | gmean | acc |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| australian | 0 | B0 | baseline | 88 | 0.479 | 0.556 | 0.699 | 0.667 | 0.699 | 0.702 |
| australian | 0 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| australian | 0 | SU1 | 1.5 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| australian | 0 | SU1 | 2.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| australian | 0 | SU1 | 2.5 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| australian | 0 | SU1 | 3.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| australian | 0 | SU2 | median_parent_le5 | 141 | 0.944 | 0.940 | 0.462 | 0.214 | 0.348 | 0.577 |
| australian | 0 | SU3 | p25 | 138 | 0.856 | 0.791 | 0.631 | 0.521 | 0.597 | 0.663 |
| australian | 0 | SU3 | p50 | 98 | 0.907 | 0.918 | 0.553 | 0.450 | 0.532 | 0.577 |
| australian | 0 | SU3 | p75 | 58 | 0.958 | 0.989 | 0.561 | 0.476 | 0.548 | 0.577 |
| australian | 1 | B0 | baseline | 97 | 0.544 | 0.608 | 0.445 | 0.648 | 0.371 | 0.519 |
| australian | 1 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| australian | 1 | SU1 | 1.5 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| australian | 1 | SU1 | 2.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| australian | 1 | SU1 | 2.5 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| australian | 1 | SU1 | 3.0 | 132 | 0.860 | 0.922 | 0.476 | 0.642 | 0.426 | 0.529 |
| australian | 1 | SU2 | median_parent_le5 | 138 | 0.912 | 0.940 | 0.474 | 0.611 | 0.441 | 0.510 |
| australian | 1 | SU3 | p25 | 131 | 0.805 | 0.847 | 0.426 | 0.603 | 0.372 | 0.481 |
| australian | 1 | SU3 | p50 | 93 | 0.912 | 0.914 | 0.328 | 0.567 | 0.212 | 0.413 |
| australian | 1 | SU3 | p75 | 63 | 0.977 | 0.978 | 0.312 | 0.563 | 0.173 | 0.404 |
| australian | 42 | B0 | baseline | 83 | 0.460 | 0.608 | 0.448 | 0.618 | 0.397 | 0.500 |
| australian | 42 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| australian | 42 | SU1 | 1.5 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| australian | 42 | SU1 | 2.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| australian | 42 | SU1 | 2.5 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| australian | 42 | SU1 | 3.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| australian | 42 | SU2 | median_parent_le5 | 141 | 0.935 | 0.944 | 0.454 | 0.595 | 0.419 | 0.490 |
| australian | 42 | SU3 | p25 | 121 | 0.833 | 0.881 | 0.379 | 0.606 | 0.284 | 0.462 |
| australian | 42 | SU3 | p50 | 95 | 0.870 | 0.910 | 0.368 | 0.611 | 0.257 | 0.462 |
| australian | 42 | SU3 | p75 | 59 | 0.944 | 0.966 | 0.368 | 0.611 | 0.257 | 0.462 |
| breast_cancer | 0 | B0 | baseline | 35 | 0.322 | 0.624 | 0.411 | 0.000 | 0.000 | 0.698 |
| breast_cancer | 0 | SU1 | 1.0 | 37 | 1.000 | 1.000 | 0.411 | 0.000 | 0.000 | 0.698 |
| breast_cancer | 0 | SU1 | 1.5 | 42 | 1.000 | 1.000 | 0.411 | 0.000 | 0.000 | 0.698 |
| breast_cancer | 0 | SU1 | 2.0 | 60 | 0.983 | 0.972 | 0.411 | 0.000 | 0.000 | 0.698 |
| breast_cancer | 0 | SU1 | 2.5 | 60 | 0.966 | 0.965 | 0.411 | 0.000 | 0.000 | 0.698 |
| breast_cancer | 0 | SU1 | 3.0 | 58 | 0.881 | 0.901 | 0.411 | 0.000 | 0.000 | 0.698 |
| breast_cancer | 0 | SU2 | median_parent_le5 | 58 | 0.966 | 0.957 | 0.411 | 0.000 | 0.000 | 0.698 |
| breast_cancer | 0 | SU3 | p25 | 60 | 0.780 | 0.872 | 0.411 | 0.000 | 0.000 | 0.698 |
| breast_cancer | 0 | SU3 | p50 | 33 | 0.966 | 0.972 | 0.411 | 0.000 | 0.000 | 0.698 |
| breast_cancer | 0 | SU3 | p75 | 16 | 0.966 | 0.979 | 0.523 | 0.235 | 0.379 | 0.698 |
| breast_cancer | 1 | B0 | baseline | 37 | 0.356 | 0.610 | 0.411 | 0.000 | 0.000 | 0.698 |
| breast_cancer | 1 | SU1 | 1.0 | 27 | 1.000 | 1.000 | 0.411 | 0.000 | 0.000 | 0.698 |
| breast_cancer | 1 | SU1 | 1.5 | 35 | 1.000 | 1.000 | 0.411 | 0.000 | 0.000 | 0.698 |
| breast_cancer | 1 | SU1 | 2.0 | 41 | 0.949 | 0.965 | 0.411 | 0.000 | 0.000 | 0.698 |
| breast_cancer | 1 | SU1 | 2.5 | 45 | 0.915 | 0.922 | 0.411 | 0.000 | 0.000 | 0.698 |
| breast_cancer | 1 | SU1 | 3.0 | 51 | 0.864 | 0.872 | 0.411 | 0.000 | 0.000 | 0.698 |
| breast_cancer | 1 | SU2 | median_parent_le5 | 48 | 0.966 | 0.887 | 0.411 | 0.000 | 0.000 | 0.698 |
| breast_cancer | 1 | SU3 | p25 | 48 | 0.763 | 0.858 | 0.411 | 0.000 | 0.000 | 0.698 |
| breast_cancer | 1 | SU3 | p50 | 26 | 0.864 | 0.908 | 0.411 | 0.000 | 0.000 | 0.698 |
| breast_cancer | 1 | SU3 | p75 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| breast_cancer | 42 | B0 | baseline | 43 | 0.339 | 0.730 | 0.411 | 0.000 | 0.000 | 0.698 |
| breast_cancer | 42 | SU1 | 1.0 | 33 | 1.000 | 1.000 | 0.411 | 0.000 | 0.000 | 0.698 |
| breast_cancer | 42 | SU1 | 1.5 | 47 | 1.000 | 0.993 | 0.411 | 0.000 | 0.000 | 0.698 |
| breast_cancer | 42 | SU1 | 2.0 | 54 | 0.949 | 0.972 | 0.411 | 0.000 | 0.000 | 0.698 |
| breast_cancer | 42 | SU1 | 2.5 | 57 | 0.932 | 0.965 | 0.411 | 0.000 | 0.000 | 0.698 |
| breast_cancer | 42 | SU1 | 3.0 | 56 | 0.814 | 0.929 | 0.411 | 0.000 | 0.000 | 0.698 |
| breast_cancer | 42 | SU2 | median_parent_le5 | 54 | 0.949 | 0.950 | 0.411 | 0.000 | 0.000 | 0.698 |
| breast_cancer | 42 | SU3 | p25 | 50 | 0.847 | 0.901 | 0.411 | 0.000 | 0.000 | 0.698 |
| breast_cancer | 42 | SU3 | p50 | 24 | 0.847 | 0.993 | 0.411 | 0.000 | 0.000 | 0.698 |
| breast_cancer | 42 | SU3 | p75 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| breast_cancer_wisconsin_diagnostic | 0 | B0 | baseline | 20 | 0.858 | 0.924 | 0.722 | 0.667 | 0.730 | 0.733 |
| breast_cancer_wisconsin_diagnostic | 0 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| breast_cancer_wisconsin_diagnostic | 0 | SU1 | 1.5 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| breast_cancer_wisconsin_diagnostic | 0 | SU1 | 2.0 | 33 | 0.980 | 1.000 | 0.599 | 0.553 | 0.614 | 0.605 |
| breast_cancer_wisconsin_diagnostic | 0 | SU1 | 2.5 | 26 | 0.980 | 0.992 | 0.787 | 0.730 | 0.782 | 0.802 |
| breast_cancer_wisconsin_diagnostic | 0 | SU1 | 3.0 | 30 | 0.953 | 0.988 | 0.763 | 0.730 | 0.781 | 0.767 |
| breast_cancer_wisconsin_diagnostic | 0 | SU2 | median_parent_le5 | 33 | 0.986 | 1.000 | 0.623 | 0.579 | 0.638 | 0.628 |
| breast_cancer_wisconsin_diagnostic | 0 | SU3 | p25 | 31 | 0.946 | 0.976 | 0.798 | 0.767 | 0.815 | 0.802 |
| breast_cancer_wisconsin_diagnostic | 0 | SU3 | p50 | 26 | 0.959 | 0.988 | 0.762 | 0.698 | 0.757 | 0.779 |
| breast_cancer_wisconsin_diagnostic | 0 | SU3 | p75 | 14 | 0.993 | 1.000 | 0.787 | 0.757 | 0.805 | 0.791 |
| breast_cancer_wisconsin_diagnostic | 1 | B0 | baseline | 20 | 0.885 | 0.948 | 0.788 | 0.763 | 0.809 | 0.791 |
| breast_cancer_wisconsin_diagnostic | 1 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| breast_cancer_wisconsin_diagnostic | 1 | SU1 | 1.5 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| breast_cancer_wisconsin_diagnostic | 1 | SU1 | 2.0 | 17 | 1.000 | 1.000 | 0.673 | 0.609 | 0.680 | 0.686 |
| breast_cancer_wisconsin_diagnostic | 1 | SU1 | 2.5 | 17 | 1.000 | 0.992 | 0.659 | 0.623 | 0.677 | 0.663 |
| breast_cancer_wisconsin_diagnostic | 1 | SU1 | 3.0 | 26 | 0.946 | 0.976 | 0.722 | 0.667 | 0.730 | 0.733 |
| breast_cancer_wisconsin_diagnostic | 1 | SU2 | median_parent_le5 | 26 | 0.980 | 0.992 | 0.739 | 0.703 | 0.756 | 0.744 |
| breast_cancer_wisconsin_diagnostic | 1 | SU3 | p25 | 18 | 0.980 | 0.988 | 0.687 | 0.629 | 0.696 | 0.698 |
| breast_cancer_wisconsin_diagnostic | 1 | SU3 | p50 | 16 | 1.000 | 1.000 | 0.546 | 0.541 | 0.565 | 0.547 |
| breast_cancer_wisconsin_diagnostic | 1 | SU3 | p75 | 12 | 1.000 | 1.000 | 0.581 | 0.591 | 0.601 | 0.581 |
| breast_cancer_wisconsin_diagnostic | 42 | B0 | baseline | 23 | 0.865 | 0.960 | 0.778 | 0.759 | 0.801 | 0.779 |
| breast_cancer_wisconsin_diagnostic | 42 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| breast_cancer_wisconsin_diagnostic | 42 | SU1 | 1.5 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| breast_cancer_wisconsin_diagnostic | 42 | SU1 | 2.0 | 24 | 0.993 | 1.000 | 0.601 | 0.564 | 0.618 | 0.605 |
| breast_cancer_wisconsin_diagnostic | 42 | SU1 | 2.5 | 24 | 0.993 | 1.000 | 0.601 | 0.564 | 0.618 | 0.605 |
| breast_cancer_wisconsin_diagnostic | 42 | SU1 | 3.0 | 29 | 0.966 | 0.984 | 0.843 | 0.812 | 0.854 | 0.849 |
| breast_cancer_wisconsin_diagnostic | 42 | SU2 | median_parent_le5 | 28 | 0.993 | 0.992 | 0.902 | 0.879 | 0.907 | 0.907 |
| breast_cancer_wisconsin_diagnostic | 42 | SU3 | p25 | 29 | 0.919 | 0.984 | 0.831 | 0.800 | 0.844 | 0.837 |
| breast_cancer_wisconsin_diagnostic | 42 | SU3 | p50 | 23 | 0.980 | 1.000 | 0.692 | 0.649 | 0.707 | 0.698 |
| breast_cancer_wisconsin_diagnostic | 42 | SU3 | p75 | 14 | 1.000 | 0.996 | 0.592 | 0.568 | 0.610 | 0.593 |
| breast_cancer_wisconsin_original | 0 | B0 | baseline | 2 | 1.000 | 1.000 | 0.894 | 0.861 | 0.894 | 0.905 |
| breast_cancer_wisconsin_original | 0 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| breast_cancer_wisconsin_original | 0 | SU1 | 1.5 | 2 | 1.000 | 1.000 | 0.894 | 0.861 | 0.894 | 0.905 |
| breast_cancer_wisconsin_original | 0 | SU1 | 2.0 | 2 | 1.000 | 1.000 | 0.894 | 0.861 | 0.894 | 0.905 |
| breast_cancer_wisconsin_original | 0 | SU1 | 2.5 | 2 | 1.000 | 1.000 | 0.894 | 0.861 | 0.894 | 0.905 |
| breast_cancer_wisconsin_original | 0 | SU1 | 3.0 | 2 | 1.000 | 1.000 | 0.894 | 0.861 | 0.894 | 0.905 |
| breast_cancer_wisconsin_original | 0 | SU2 | median_parent_le5 | 2 | 1.000 | 1.000 | 0.894 | 0.861 | 0.894 | 0.905 |
| breast_cancer_wisconsin_original | 0 | SU3 | p25 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| breast_cancer_wisconsin_original | 0 | SU3 | p50 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| breast_cancer_wisconsin_original | 0 | SU3 | p75 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| breast_cancer_wisconsin_original | 1 | B0 | baseline | 2 | 1.000 | 1.000 | 0.835 | 0.789 | 0.844 | 0.848 |
| breast_cancer_wisconsin_original | 1 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| breast_cancer_wisconsin_original | 1 | SU1 | 1.5 | 2 | 1.000 | 1.000 | 0.835 | 0.789 | 0.844 | 0.848 |
| breast_cancer_wisconsin_original | 1 | SU1 | 2.0 | 2 | 1.000 | 1.000 | 0.835 | 0.789 | 0.844 | 0.848 |
| breast_cancer_wisconsin_original | 1 | SU1 | 2.5 | 2 | 1.000 | 1.000 | 0.835 | 0.789 | 0.844 | 0.848 |
| breast_cancer_wisconsin_original | 1 | SU1 | 3.0 | 2 | 1.000 | 1.000 | 0.835 | 0.789 | 0.844 | 0.848 |
| breast_cancer_wisconsin_original | 1 | SU2 | median_parent_le5 | 2 | 1.000 | 1.000 | 0.835 | 0.789 | 0.844 | 0.848 |
| breast_cancer_wisconsin_original | 1 | SU3 | p25 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| breast_cancer_wisconsin_original | 1 | SU3 | p50 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| breast_cancer_wisconsin_original | 1 | SU3 | p75 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| breast_cancer_wisconsin_original | 42 | B0 | baseline | 2 | 1.000 | 1.000 | 0.816 | 0.769 | 0.830 | 0.829 |
| breast_cancer_wisconsin_original | 42 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| breast_cancer_wisconsin_original | 42 | SU1 | 1.5 | 2 | 1.000 | 1.000 | 0.816 | 0.769 | 0.830 | 0.829 |
| breast_cancer_wisconsin_original | 42 | SU1 | 2.0 | 2 | 1.000 | 1.000 | 0.816 | 0.769 | 0.830 | 0.829 |
| breast_cancer_wisconsin_original | 42 | SU1 | 2.5 | 2 | 1.000 | 1.000 | 0.816 | 0.769 | 0.830 | 0.829 |
| breast_cancer_wisconsin_original | 42 | SU1 | 3.0 | 2 | 1.000 | 1.000 | 0.816 | 0.769 | 0.830 | 0.829 |
| breast_cancer_wisconsin_original | 42 | SU2 | median_parent_le5 | 2 | 1.000 | 1.000 | 0.816 | 0.769 | 0.830 | 0.829 |
| breast_cancer_wisconsin_original | 42 | SU3 | p25 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| breast_cancer_wisconsin_original | 42 | SU3 | p50 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| breast_cancer_wisconsin_original | 42 | SU3 | p75 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| chronic_kidney_disease | 0 | B0 | baseline | 52 | 0.686 | 0.640 | 0.388 | 0.000 | 0.000 | 0.633 |
| chronic_kidney_disease | 0 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| chronic_kidney_disease | 0 | SU1 | 1.5 | 2 | 1.000 | 1.000 | nan | nan | nan | nan |
| chronic_kidney_disease | 0 | SU1 | 2.0 | 11 | 1.000 | 1.000 | 0.600 | 0.586 | 0.622 | 0.600 |
| chronic_kidney_disease | 0 | SU1 | 2.5 | 35 | 0.971 | 0.989 | 0.479 | 0.167 | 0.302 | 0.667 |
| chronic_kidney_disease | 0 | SU1 | 3.0 | 39 | 0.981 | 0.971 | 0.427 | 0.083 | 0.210 | 0.633 |
| chronic_kidney_disease | 0 | SU2 | median_parent_le5 | 75 | 0.943 | 0.920 | 0.388 | 0.000 | 0.000 | 0.633 |
| chronic_kidney_disease | 0 | SU3 | p25 | 63 | 0.943 | 0.834 | 0.388 | 0.000 | 0.000 | 0.633 |
| chronic_kidney_disease | 0 | SU3 | p50 | 54 | 1.000 | 0.909 | 0.388 | 0.000 | 0.000 | 0.633 |
| chronic_kidney_disease | 0 | SU3 | p75 | 33 | 1.000 | 0.989 | 0.411 | 0.077 | 0.205 | 0.600 |
| chronic_kidney_disease | 1 | B0 | baseline | 51 | 0.600 | 0.629 | 0.427 | 0.083 | 0.209 | 0.633 |
| chronic_kidney_disease | 1 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| chronic_kidney_disease | 1 | SU1 | 1.5 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| chronic_kidney_disease | 1 | SU1 | 2.0 | 4 | 1.000 | 1.000 | 0.437 | 0.522 | 0.436 | 0.450 |
| chronic_kidney_disease | 1 | SU1 | 2.5 | 23 | 0.981 | 1.000 | 0.467 | 0.200 | 0.341 | 0.600 |
| chronic_kidney_disease | 1 | SU1 | 3.0 | 59 | 0.962 | 0.954 | 0.381 | 0.000 | 0.000 | 0.617 |
| chronic_kidney_disease | 1 | SU2 | median_parent_le5 | 66 | 0.971 | 0.891 | 0.381 | 0.000 | 0.000 | 0.617 |
| chronic_kidney_disease | 1 | SU3 | p25 | 56 | 0.848 | 0.817 | 0.368 | 0.000 | 0.000 | 0.583 |
| chronic_kidney_disease | 1 | SU3 | p50 | 51 | 0.943 | 0.869 | 0.368 | 0.000 | 0.000 | 0.583 |
| chronic_kidney_disease | 1 | SU3 | p75 | 35 | 1.000 | 0.971 | 0.466 | 0.235 | 0.376 | 0.567 |
| chronic_kidney_disease | 42 | B0 | baseline | 52 | 0.657 | 0.663 | 0.479 | 0.167 | 0.302 | 0.667 |
| chronic_kidney_disease | 42 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| chronic_kidney_disease | 42 | SU1 | 1.5 | 2 | 1.000 | 1.000 | 0.588 | 0.450 | 0.559 | 0.633 |
| chronic_kidney_disease | 42 | SU1 | 2.0 | 2 | 1.000 | 1.000 | 0.588 | 0.450 | 0.559 | 0.633 |
| chronic_kidney_disease | 42 | SU1 | 2.5 | 52 | 1.000 | 0.960 | 0.479 | 0.167 | 0.302 | 0.667 |
| chronic_kidney_disease | 42 | SU1 | 3.0 | 57 | 0.952 | 0.914 | 0.388 | 0.000 | 0.000 | 0.633 |
| chronic_kidney_disease | 42 | SU2 | median_parent_le5 | 65 | 0.962 | 0.909 | 0.381 | 0.000 | 0.000 | 0.617 |
| chronic_kidney_disease | 42 | SU3 | p25 | 60 | 0.924 | 0.829 | 0.435 | 0.087 | 0.213 | 0.650 |
| chronic_kidney_disease | 42 | SU3 | p50 | 45 | 0.971 | 0.909 | 0.375 | 0.000 | 0.000 | 0.600 |
| chronic_kidney_disease | 42 | SU3 | p75 | 31 | 1.000 | 0.994 | 0.450 | 0.148 | 0.289 | 0.617 |
| congressional_voting_records | 0 | B0 | baseline | 8 | 0.974 | 0.984 | 0.937 | 0.923 | 0.936 | 0.939 |
| congressional_voting_records | 0 | SU1 | 1.0 | 9 | 1.000 | 1.000 | 0.937 | 0.926 | 0.943 | 0.939 |
| congressional_voting_records | 0 | SU1 | 1.5 | 9 | 1.000 | 1.000 | 0.937 | 0.926 | 0.943 | 0.939 |
| congressional_voting_records | 0 | SU1 | 2.0 | 9 | 1.000 | 1.000 | 0.937 | 0.926 | 0.943 | 0.939 |
| congressional_voting_records | 0 | SU1 | 2.5 | 9 | 0.991 | 1.000 | 0.937 | 0.926 | 0.943 | 0.939 |
| congressional_voting_records | 0 | SU1 | 3.0 | 9 | 0.991 | 0.995 | 0.921 | 0.906 | 0.924 | 0.924 |
| congressional_voting_records | 0 | SU2 | median_parent_le5 | 9 | 1.000 | 1.000 | 0.937 | 0.926 | 0.943 | 0.939 |
| congressional_voting_records | 0 | SU3 | p25 | 2 | 1.000 | 1.000 | 0.903 | 0.880 | 0.897 | 0.909 |
| congressional_voting_records | 0 | SU3 | p50 | 2 | 1.000 | 1.000 | 0.903 | 0.880 | 0.897 | 0.909 |
| congressional_voting_records | 0 | SU3 | p75 | 2 | 1.000 | 1.000 | 0.903 | 0.880 | 0.897 | 0.909 |
| congressional_voting_records | 1 | B0 | baseline | 30 | 0.897 | 0.920 | 0.937 | 0.923 | 0.936 | 0.939 |
| congressional_voting_records | 1 | SU1 | 1.0 | 23 | 1.000 | 1.000 | 0.907 | 0.893 | 0.917 | 0.909 |
| congressional_voting_records | 1 | SU1 | 1.5 | 27 | 1.000 | 0.995 | 0.906 | 0.889 | 0.911 | 0.909 |
| congressional_voting_records | 1 | SU1 | 2.0 | 33 | 0.983 | 0.989 | 0.833 | 0.783 | 0.811 | 0.848 |
| congressional_voting_records | 1 | SU1 | 2.5 | 31 | 0.974 | 0.963 | 0.854 | 0.816 | 0.844 | 0.864 |
| congressional_voting_records | 1 | SU1 | 3.0 | 34 | 0.932 | 0.963 | 0.848 | 0.800 | 0.822 | 0.864 |
| congressional_voting_records | 1 | SU2 | median_parent_le5 | 33 | 1.000 | 0.968 | 0.738 | 0.667 | 0.723 | 0.758 |
| congressional_voting_records | 1 | SU3 | p25 | 13 | 1.000 | 0.989 | 0.829 | 0.800 | 0.836 | 0.833 |
| congressional_voting_records | 1 | SU3 | p50 | 2 | 1.000 | 1.000 | 0.875 | 0.852 | 0.880 | 0.879 |
| congressional_voting_records | 1 | SU3 | p75 | 2 | 1.000 | 1.000 | 0.875 | 0.852 | 0.880 | 0.879 |
| congressional_voting_records | 42 | B0 | baseline | 24 | 0.889 | 0.914 | 0.741 | 0.632 | 0.679 | 0.788 |
| congressional_voting_records | 42 | SU1 | 1.0 | 26 | 1.000 | 1.000 | 0.377 | 0.000 | 0.000 | 0.606 |
| congressional_voting_records | 42 | SU1 | 1.5 | 32 | 1.000 | 0.984 | 0.377 | 0.000 | 0.000 | 0.606 |
| congressional_voting_records | 42 | SU1 | 2.0 | 29 | 0.974 | 0.989 | 0.377 | 0.000 | 0.000 | 0.606 |
| congressional_voting_records | 42 | SU1 | 2.5 | 30 | 0.949 | 0.968 | 0.377 | 0.000 | 0.000 | 0.606 |
| congressional_voting_records | 42 | SU1 | 3.0 | 31 | 0.949 | 0.973 | 0.456 | 0.143 | 0.277 | 0.636 |
| congressional_voting_records | 42 | SU2 | median_parent_le5 | 30 | 0.991 | 1.000 | 0.377 | 0.000 | 0.000 | 0.606 |
| congressional_voting_records | 42 | SU3 | p25 | 26 | 0.966 | 0.979 | 0.377 | 0.000 | 0.000 | 0.606 |
| congressional_voting_records | 42 | SU3 | p50 | 3 | 1.000 | 1.000 | 0.937 | 0.926 | 0.943 | 0.939 |
| congressional_voting_records | 42 | SU3 | p75 | 2 | 1.000 | 1.000 | 0.846 | 0.828 | 0.859 | 0.848 |
| credit_approval | 0 | B0 | baseline | 99 | 0.516 | 0.638 | 0.380 | 0.630 | 0.263 | 0.481 |
| credit_approval | 0 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| credit_approval | 0 | SU1 | 1.5 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| credit_approval | 0 | SU1 | 2.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| credit_approval | 0 | SU1 | 2.5 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| credit_approval | 0 | SU1 | 3.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| credit_approval | 0 | SU2 | median_parent_le5 | 143 | 0.935 | 0.929 | 0.302 | 0.604 | 0.000 | 0.433 |
| credit_approval | 0 | SU3 | p25 | 120 | 0.847 | 0.892 | 0.297 | 0.595 | 0.000 | 0.423 |
| credit_approval | 0 | SU3 | p50 | 104 | 0.893 | 0.929 | 0.307 | 0.613 | 0.000 | 0.442 |
| credit_approval | 0 | SU3 | p75 | 1 | 0.995 | 1.000 | nan | nan | nan | nan |
| credit_approval | 1 | B0 | baseline | 97 | 0.544 | 0.627 | 0.326 | 0.617 | 0.131 | 0.452 |
| credit_approval | 1 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| credit_approval | 1 | SU1 | 1.5 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| credit_approval | 1 | SU1 | 2.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| credit_approval | 1 | SU1 | 2.5 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| credit_approval | 1 | SU1 | 3.0 | 105 | 0.916 | 0.910 | 0.307 | 0.613 | 0.000 | 0.442 |
| credit_approval | 1 | SU2 | median_parent_le5 | 145 | 0.935 | 0.929 | 0.307 | 0.613 | 0.000 | 0.442 |
| credit_approval | 1 | SU3 | p25 | 129 | 0.833 | 0.854 | 0.307 | 0.613 | 0.000 | 0.442 |
| credit_approval | 1 | SU3 | p50 | 93 | 0.916 | 0.933 | 0.326 | 0.617 | 0.131 | 0.452 |
| credit_approval | 1 | SU3 | p75 | 48 | 0.963 | 0.970 | 0.307 | 0.613 | 0.000 | 0.442 |
| credit_approval | 42 | B0 | baseline | 82 | 0.474 | 0.612 | 0.302 | 0.604 | 0.000 | 0.433 |
| credit_approval | 42 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| credit_approval | 42 | SU1 | 1.5 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| credit_approval | 42 | SU1 | 2.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| credit_approval | 42 | SU1 | 2.5 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| credit_approval | 42 | SU1 | 3.0 | 131 | 0.865 | 0.910 | 0.307 | 0.613 | 0.000 | 0.442 |
| credit_approval | 42 | SU2 | median_parent_le5 | 142 | 0.944 | 0.933 | 0.344 | 0.622 | 0.186 | 0.462 |
| credit_approval | 42 | SU3 | p25 | 142 | 0.819 | 0.862 | 0.302 | 0.604 | 0.000 | 0.433 |
| credit_approval | 42 | SU3 | p50 | 95 | 0.916 | 0.925 | 0.307 | 0.613 | 0.000 | 0.442 |
| credit_approval | 42 | SU3 | p75 | 54 | 0.972 | 0.951 | 0.302 | 0.604 | 0.000 | 0.433 |
| diabetes | 0 | B0 | baseline | 84 | 0.460 | 0.700 | 0.393 | 0.000 | 0.000 | 0.647 |
| diabetes | 0 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| diabetes | 0 | SU1 | 1.5 | 73 | 1.000 | 0.997 | 0.393 | 0.000 | 0.000 | 0.647 |
| diabetes | 0 | SU1 | 2.0 | 82 | 0.989 | 0.980 | 0.393 | 0.000 | 0.000 | 0.647 |
| diabetes | 0 | SU1 | 2.5 | 129 | 0.893 | 0.946 | 0.393 | 0.000 | 0.000 | 0.647 |
| diabetes | 0 | SU1 | 3.0 | 131 | 0.850 | 0.906 | 0.393 | 0.000 | 0.000 | 0.647 |
| diabetes | 0 | SU2 | median_parent_le5 | 127 | 0.925 | 0.954 | 0.393 | 0.000 | 0.000 | 0.647 |
| diabetes | 0 | SU3 | p25 | 129 | 0.791 | 0.883 | 0.393 | 0.000 | 0.000 | 0.647 |
| diabetes | 0 | SU3 | p50 | 97 | 0.941 | 0.960 | 0.393 | 0.000 | 0.000 | 0.647 |
| diabetes | 0 | SU3 | p75 | 28 | 0.984 | 0.986 | 0.411 | 0.045 | 0.154 | 0.638 |
| diabetes | 1 | B0 | baseline | 86 | 0.439 | 0.689 | 0.393 | 0.000 | 0.000 | 0.647 |
| diabetes | 1 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| diabetes | 1 | SU1 | 1.5 | 113 | 0.995 | 1.000 | 0.393 | 0.000 | 0.000 | 0.647 |
| diabetes | 1 | SU1 | 2.0 | 106 | 0.947 | 0.991 | 0.393 | 0.000 | 0.000 | 0.647 |
| diabetes | 1 | SU1 | 2.5 | 124 | 0.893 | 0.949 | 0.393 | 0.000 | 0.000 | 0.647 |
| diabetes | 1 | SU1 | 3.0 | 132 | 0.850 | 0.891 | 0.393 | 0.000 | 0.000 | 0.647 |
| diabetes | 1 | SU2 | median_parent_le5 | 134 | 0.968 | 0.963 | 0.393 | 0.000 | 0.000 | 0.647 |
| diabetes | 1 | SU3 | p25 | 127 | 0.807 | 0.874 | 0.393 | 0.000 | 0.000 | 0.647 |
| diabetes | 1 | SU3 | p50 | 85 | 0.909 | 0.946 | 0.393 | 0.000 | 0.000 | 0.647 |
| diabetes | 1 | SU3 | p75 | 45 | 0.984 | 0.991 | 0.393 | 0.000 | 0.000 | 0.647 |
| diabetes | 42 | B0 | baseline | 85 | 0.449 | 0.691 | 0.393 | 0.000 | 0.000 | 0.647 |
| diabetes | 42 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| diabetes | 42 | SU1 | 1.5 | 73 | 1.000 | 1.000 | 0.393 | 0.000 | 0.000 | 0.647 |
| diabetes | 42 | SU1 | 2.0 | 117 | 0.979 | 0.971 | 0.393 | 0.000 | 0.000 | 0.647 |
| diabetes | 42 | SU1 | 2.5 | 108 | 0.872 | 0.931 | 0.393 | 0.000 | 0.000 | 0.647 |
| diabetes | 42 | SU1 | 3.0 | 123 | 0.861 | 0.903 | 0.393 | 0.000 | 0.000 | 0.647 |
| diabetes | 42 | SU2 | median_parent_le5 | 137 | 0.898 | 0.960 | 0.393 | 0.000 | 0.000 | 0.647 |
| diabetes | 42 | SU3 | p25 | 118 | 0.765 | 0.911 | 0.393 | 0.000 | 0.000 | 0.647 |
| diabetes | 42 | SU3 | p50 | 80 | 0.904 | 0.963 | 0.393 | 0.000 | 0.000 | 0.647 |
| diabetes | 42 | SU3 | p75 | 46 | 0.957 | 0.994 | 0.393 | 0.000 | 0.000 | 0.647 |
| ecoli2 | 0 | B0 | baseline | 15 | 0.861 | 0.970 | 0.452 | 0.000 | 0.000 | 0.824 |
| ecoli2 | 0 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| ecoli2 | 0 | SU1 | 1.5 | 18 | 1.000 | 1.000 | 0.457 | 0.000 | 0.000 | 0.843 |
| ecoli2 | 0 | SU1 | 2.0 | 19 | 0.972 | 1.000 | 0.457 | 0.000 | 0.000 | 0.843 |
| ecoli2 | 0 | SU1 | 2.5 | 19 | 0.972 | 0.995 | 0.457 | 0.000 | 0.000 | 0.843 |
| ecoli2 | 0 | SU1 | 3.0 | 19 | 0.972 | 0.990 | 0.457 | 0.000 | 0.000 | 0.843 |
| ecoli2 | 0 | SU2 | median_parent_le5 | 17 | 0.972 | 1.000 | 0.457 | 0.000 | 0.000 | 0.843 |
| ecoli2 | 0 | SU3 | p25 | 18 | 1.000 | 1.000 | 0.457 | 0.000 | 0.000 | 0.843 |
| ecoli2 | 0 | SU3 | p50 | 3 | 1.000 | 1.000 | 0.790 | 0.667 | 0.868 | 0.863 |
| ecoli2 | 0 | SU3 | p75 | 3 | 1.000 | 1.000 | 0.790 | 0.667 | 0.868 | 0.863 |
| ecoli2 | 1 | B0 | baseline | 17 | 0.833 | 0.975 | 0.457 | 0.000 | 0.000 | 0.843 |
| ecoli2 | 1 | SU1 | 1.0 | 3 | 1.000 | 1.000 | 0.865 | 0.778 | 0.902 | 0.922 |
| ecoli2 | 1 | SU1 | 1.5 | 17 | 1.000 | 0.995 | 0.457 | 0.000 | 0.000 | 0.843 |
| ecoli2 | 1 | SU1 | 2.0 | 17 | 0.972 | 0.995 | 0.457 | 0.000 | 0.000 | 0.843 |
| ecoli2 | 1 | SU1 | 2.5 | 19 | 0.917 | 0.995 | 0.457 | 0.000 | 0.000 | 0.843 |
| ecoli2 | 1 | SU1 | 3.0 | 19 | 0.889 | 0.995 | 0.457 | 0.000 | 0.000 | 0.843 |
| ecoli2 | 1 | SU2 | median_parent_le5 | 17 | 0.972 | 0.995 | 0.457 | 0.000 | 0.000 | 0.843 |
| ecoli2 | 1 | SU3 | p25 | 3 | 1.000 | 1.000 | 0.865 | 0.778 | 0.902 | 0.922 |
| ecoli2 | 1 | SU3 | p50 | 3 | 1.000 | 1.000 | 0.865 | 0.778 | 0.902 | 0.922 |
| ecoli2 | 1 | SU3 | p75 | 3 | 1.000 | 1.000 | 0.865 | 0.778 | 0.902 | 0.922 |
| ecoli2 | 42 | B0 | baseline | 13 | 0.750 | 0.955 | 0.805 | 0.667 | 0.772 | 0.902 |
| ecoli2 | 42 | SU1 | 1.0 | 7 | 1.000 | 1.000 | 0.423 | 0.356 | 0.571 | 0.431 |
| ecoli2 | 42 | SU1 | 1.5 | 18 | 1.000 | 1.000 | 0.457 | 0.000 | 0.000 | 0.843 |
| ecoli2 | 42 | SU1 | 2.0 | 18 | 0.944 | 1.000 | 0.457 | 0.000 | 0.000 | 0.843 |
| ecoli2 | 42 | SU1 | 2.5 | 18 | 0.944 | 0.990 | 0.457 | 0.000 | 0.000 | 0.843 |
| ecoli2 | 42 | SU1 | 3.0 | 18 | 0.944 | 0.980 | 0.457 | 0.000 | 0.000 | 0.843 |
| ecoli2 | 42 | SU2 | median_parent_le5 | 18 | 1.000 | 1.000 | 0.457 | 0.000 | 0.000 | 0.843 |
| ecoli2 | 42 | SU3 | p25 | 18 | 0.944 | 0.985 | 0.457 | 0.000 | 0.000 | 0.843 |
| ecoli2 | 42 | SU3 | p50 | 6 | 1.000 | 1.000 | 0.606 | 0.452 | 0.741 | 0.667 |
| ecoli2 | 42 | SU3 | p75 | 6 | 1.000 | 1.000 | 0.606 | 0.452 | 0.741 | 0.667 |
| german_credit | 0 | B0 | baseline | 119 | 0.252 | 0.590 | 0.402 | 0.000 | 0.000 | 0.673 |
| german_credit | 0 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| german_credit | 0 | SU1 | 1.5 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| german_credit | 0 | SU1 | 2.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| german_credit | 0 | SU1 | 2.5 | 181 | 0.938 | 0.947 | 0.430 | 0.042 | 0.148 | 0.693 |
| german_credit | 0 | SU1 | 3.0 | 217 | 0.905 | 0.918 | 0.407 | 0.000 | 0.000 | 0.687 |
| german_credit | 0 | SU2 | median_parent_le5 | 219 | 0.938 | 0.947 | 0.407 | 0.000 | 0.000 | 0.687 |
| german_credit | 0 | SU3 | p25 | 214 | 0.786 | 0.824 | 0.432 | 0.043 | 0.148 | 0.700 |
| german_credit | 0 | SU3 | p50 | 173 | 0.919 | 0.935 | 0.409 | 0.000 | 0.000 | 0.693 |
| german_credit | 0 | SU3 | p75 | 110 | 0.986 | 0.992 | 0.451 | 0.082 | 0.209 | 0.700 |
| german_credit | 1 | B0 | baseline | 130 | 0.376 | 0.635 | 0.409 | 0.000 | 0.000 | 0.693 |
| german_credit | 1 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| german_credit | 1 | SU1 | 1.5 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| german_credit | 1 | SU1 | 2.0 | 12 | 1.000 | 1.000 | nan | nan | nan | nan |
| german_credit | 1 | SU1 | 2.5 | 171 | 0.933 | 0.945 | 0.458 | 0.111 | 0.251 | 0.680 |
| german_credit | 1 | SU1 | 3.0 | 195 | 0.943 | 0.896 | 0.405 | 0.000 | 0.000 | 0.680 |
| german_credit | 1 | SU2 | median_parent_le5 | 202 | 0.962 | 0.957 | 0.427 | 0.041 | 0.147 | 0.687 |
| german_credit | 1 | SU3 | p25 | 207 | 0.814 | 0.839 | 0.407 | 0.000 | 0.000 | 0.687 |
| german_credit | 1 | SU3 | p50 | 167 | 0.890 | 0.957 | 0.435 | 0.043 | 0.149 | 0.707 |
| german_credit | 1 | SU3 | p75 | 101 | 0.990 | 0.984 | 0.448 | 0.080 | 0.208 | 0.693 |
| german_credit | 42 | B0 | baseline | 118 | 0.229 | 0.604 | 0.409 | 0.000 | 0.000 | 0.693 |
| german_credit | 42 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| german_credit | 42 | SU1 | 1.5 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| german_credit | 42 | SU1 | 2.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| german_credit | 42 | SU1 | 2.5 | 163 | 0.952 | 0.961 | 0.430 | 0.042 | 0.148 | 0.693 |
| german_credit | 42 | SU1 | 3.0 | 196 | 0.905 | 0.929 | 0.412 | 0.000 | 0.000 | 0.700 |
| german_credit | 42 | SU2 | median_parent_le5 | 214 | 0.933 | 0.935 | 0.409 | 0.000 | 0.000 | 0.693 |
| german_credit | 42 | SU3 | p25 | 224 | 0.752 | 0.841 | 0.407 | 0.000 | 0.000 | 0.687 |
| german_credit | 42 | SU3 | p50 | 164 | 0.910 | 0.929 | 0.412 | 0.000 | 0.000 | 0.700 |
| german_credit | 42 | SU3 | p75 | 107 | 0.976 | 0.988 | 0.432 | 0.043 | 0.148 | 0.700 |
| heart | 0 | B0 | baseline | 31 | 0.440 | 0.514 | 0.529 | 0.333 | 0.450 | 0.610 |
| heart | 0 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| heart | 0 | SU1 | 1.5 | 51 | 1.000 | 1.000 | 0.359 | 0.000 | 0.000 | 0.561 |
| heart | 0 | SU1 | 2.0 | 51 | 0.976 | 0.990 | 0.359 | 0.000 | 0.000 | 0.561 |
| heart | 0 | SU1 | 2.5 | 51 | 0.905 | 0.952 | 0.359 | 0.000 | 0.000 | 0.561 |
| heart | 0 | SU1 | 3.0 | 61 | 0.821 | 0.819 | 0.359 | 0.000 | 0.000 | 0.561 |
| heart | 0 | SU2 | median_parent_le5 | 59 | 0.940 | 0.933 | 0.359 | 0.000 | 0.000 | 0.561 |
| heart | 0 | SU3 | p25 | 48 | 0.774 | 0.810 | 0.359 | 0.000 | 0.000 | 0.561 |
| heart | 0 | SU3 | p50 | 34 | 0.940 | 0.905 | 0.456 | 0.190 | 0.326 | 0.585 |
| heart | 0 | SU3 | p75 | 16 | 1.000 | 1.000 | 0.634 | 0.634 | 0.639 | 0.634 |
| heart | 1 | B0 | baseline | 34 | 0.464 | 0.486 | 0.632 | 0.533 | 0.606 | 0.659 |
| heart | 1 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| heart | 1 | SU1 | 1.5 | 41 | 1.000 | 1.000 | 0.359 | 0.000 | 0.000 | 0.561 |
| heart | 1 | SU1 | 2.0 | 57 | 0.952 | 0.952 | 0.359 | 0.000 | 0.000 | 0.561 |
| heart | 1 | SU1 | 2.5 | 65 | 0.929 | 0.952 | 0.405 | 0.100 | 0.231 | 0.561 |
| heart | 1 | SU1 | 3.0 | 62 | 0.821 | 0.838 | 0.418 | 0.105 | 0.236 | 0.585 |
| heart | 1 | SU2 | median_parent_le5 | 57 | 0.964 | 0.933 | 0.418 | 0.105 | 0.236 | 0.585 |
| heart | 1 | SU3 | p25 | 61 | 0.833 | 0.771 | 0.359 | 0.000 | 0.000 | 0.561 |
| heart | 1 | SU3 | p50 | 44 | 0.905 | 0.914 | 0.359 | 0.000 | 0.000 | 0.561 |
| heart | 1 | SU3 | p75 | 23 | 0.988 | 0.981 | 0.569 | 0.485 | 0.556 | 0.585 |
| heart | 42 | B0 | baseline | 38 | 0.548 | 0.610 | 0.549 | 0.385 | 0.491 | 0.610 |
| heart | 42 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| heart | 42 | SU1 | 1.5 | 53 | 1.000 | 1.000 | 0.520 | 0.286 | 0.408 | 0.634 |
| heart | 42 | SU1 | 2.0 | 57 | 0.988 | 0.981 | 0.359 | 0.000 | 0.000 | 0.561 |
| heart | 42 | SU1 | 2.5 | 57 | 0.952 | 0.905 | 0.471 | 0.200 | 0.333 | 0.610 |
| heart | 42 | SU1 | 3.0 | 59 | 0.833 | 0.857 | 0.359 | 0.000 | 0.000 | 0.561 |
| heart | 42 | SU2 | median_parent_le5 | 59 | 0.964 | 0.952 | 0.359 | 0.000 | 0.000 | 0.561 |
| heart | 42 | SU3 | p25 | 58 | 0.810 | 0.848 | 0.359 | 0.000 | 0.000 | 0.561 |
| heart | 42 | SU3 | p50 | 48 | 0.952 | 0.933 | 0.418 | 0.105 | 0.236 | 0.585 |
| heart | 42 | SU3 | p75 | 23 | 1.000 | 0.971 | 0.600 | 0.483 | 0.567 | 0.634 |
| ionosphere | 0 | B0 | baseline | 27 | 0.705 | 0.968 | 0.813 | 0.757 | 0.806 | 0.830 |
| ionosphere | 0 | SU1 | 1.0 | 7 | 1.000 | 1.000 | 0.667 | 0.605 | 0.680 | 0.679 |
| ionosphere | 0 | SU1 | 1.5 | 11 | 1.000 | 1.000 | 0.802 | 0.727 | 0.771 | 0.830 |
| ionosphere | 0 | SU1 | 2.0 | 11 | 1.000 | 1.000 | 0.802 | 0.727 | 0.771 | 0.830 |
| ionosphere | 0 | SU1 | 2.5 | 11 | 1.000 | 1.000 | 0.802 | 0.727 | 0.771 | 0.830 |
| ionosphere | 0 | SU1 | 3.0 | 11 | 1.000 | 1.000 | 0.802 | 0.727 | 0.771 | 0.830 |
| ionosphere | 0 | SU2 | median_parent_le5 | 34 | 0.693 | 1.000 | 0.549 | 0.345 | 0.474 | 0.642 |
| ionosphere | 0 | SU3 | p25 | 12 | 0.943 | 1.000 | 0.870 | 0.824 | 0.846 | 0.887 |
| ionosphere | 0 | SU3 | p50 | 11 | 0.886 | 0.994 | 0.783 | 0.706 | 0.759 | 0.811 |
| ionosphere | 0 | SU3 | p75 | 7 | 1.000 | 1.000 | 0.721 | 0.600 | 0.668 | 0.774 |
| ionosphere | 1 | B0 | baseline | 32 | 0.580 | 0.949 | 0.450 | 0.160 | 0.305 | 0.604 |
| ionosphere | 1 | SU1 | 1.0 | 7 | 1.000 | 1.000 | 0.840 | 0.774 | 0.795 | 0.868 |
| ionosphere | 1 | SU1 | 1.5 | 13 | 1.000 | 1.000 | 0.776 | 0.688 | 0.738 | 0.811 |
| ionosphere | 1 | SU1 | 2.0 | 13 | 0.989 | 1.000 | 0.846 | 0.788 | 0.815 | 0.868 |
| ionosphere | 1 | SU1 | 2.5 | 17 | 0.886 | 0.994 | 0.723 | 0.629 | 0.703 | 0.755 |
| ionosphere | 1 | SU1 | 3.0 | 29 | 0.739 | 0.981 | 0.677 | 0.519 | 0.598 | 0.755 |
| ionosphere | 1 | SU2 | median_parent_le5 | 36 | 0.670 | 0.994 | 0.450 | 0.160 | 0.305 | 0.604 |
| ionosphere | 1 | SU3 | p25 | 13 | 0.943 | 1.000 | 0.749 | 0.645 | 0.704 | 0.792 |
| ionosphere | 1 | SU3 | p50 | 11 | 0.966 | 0.994 | 0.814 | 0.733 | 0.761 | 0.849 |
| ionosphere | 1 | SU3 | p75 | 7 | 0.977 | 1.000 | 0.758 | 0.667 | 0.727 | 0.792 |
| ionosphere | 42 | B0 | baseline | 24 | 0.648 | 0.962 | 0.714 | 0.606 | 0.681 | 0.755 |
| ionosphere | 42 | SU1 | 1.0 | 11 | 1.000 | 1.000 | 0.637 | 0.514 | 0.613 | 0.679 |
| ionosphere | 42 | SU1 | 1.5 | 11 | 1.000 | 1.000 | 0.637 | 0.514 | 0.613 | 0.679 |
| ionosphere | 42 | SU1 | 2.0 | 15 | 0.977 | 1.000 | 0.832 | 0.778 | 0.820 | 0.849 |
| ionosphere | 42 | SU1 | 2.5 | 19 | 0.886 | 0.994 | 0.709 | 0.571 | 0.639 | 0.774 |
| ionosphere | 42 | SU1 | 3.0 | 16 | 0.852 | 0.994 | 0.704 | 0.581 | 0.657 | 0.755 |
| ionosphere | 42 | SU2 | median_parent_le5 | 23 | 0.659 | 0.994 | 0.723 | 0.629 | 0.703 | 0.755 |
| ionosphere | 42 | SU3 | p25 | 5 | 0.989 | 1.000 | 0.713 | 0.632 | 0.708 | 0.736 |
| ionosphere | 42 | SU3 | p50 | 5 | 0.989 | 1.000 | 0.713 | 0.632 | 0.708 | 0.736 |
| ionosphere | 42 | SU3 | p75 | 2 | 1.000 | 1.000 | 0.663 | 0.556 | 0.646 | 0.698 |
| messidor_features | 0 | B0 | baseline | 147 | 0.487 | 0.536 | 0.563 | 0.702 | 0.518 | 0.607 |
| messidor_features | 0 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| messidor_features | 0 | SU1 | 1.5 | 189 | 0.992 | 0.998 | 0.494 | 0.678 | 0.427 | 0.561 |
| messidor_features | 0 | SU1 | 2.0 | 230 | 0.984 | 0.972 | 0.384 | 0.648 | 0.254 | 0.497 |
| messidor_features | 0 | SU1 | 2.5 | 246 | 0.905 | 0.892 | 0.343 | 0.643 | 0.147 | 0.480 |
| messidor_features | 0 | SU1 | 3.0 | 246 | 0.886 | 0.850 | 0.430 | 0.664 | 0.330 | 0.526 |
| messidor_features | 0 | SU2 | median_parent_le5 | 240 | 0.942 | 0.937 | 0.488 | 0.681 | 0.417 | 0.561 |
| messidor_features | 0 | SU3 | p25 | 225 | 0.825 | 0.829 | 0.498 | 0.684 | 0.430 | 0.566 |
| messidor_features | 0 | SU3 | p50 | 171 | 0.929 | 0.897 | 0.430 | 0.664 | 0.330 | 0.526 |
| messidor_features | 0 | SU3 | p75 | 80 | 0.979 | 0.958 | 0.460 | 0.672 | 0.376 | 0.543 |
| messidor_features | 1 | B0 | baseline | 139 | 0.495 | 0.541 | 0.570 | 0.696 | 0.532 | 0.607 |
| messidor_features | 1 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| messidor_features | 1 | SU1 | 1.5 | 208 | 1.000 | 1.000 | 0.366 | 0.648 | 0.209 | 0.491 |
| messidor_features | 1 | SU1 | 2.0 | 209 | 0.976 | 0.972 | 0.440 | 0.667 | 0.346 | 0.532 |
| messidor_features | 1 | SU1 | 2.5 | 236 | 0.894 | 0.899 | 0.399 | 0.656 | 0.276 | 0.509 |
| messidor_features | 1 | SU1 | 3.0 | 248 | 0.870 | 0.843 | 0.388 | 0.653 | 0.255 | 0.503 |
| messidor_features | 1 | SU2 | median_parent_le5 | 240 | 0.955 | 0.951 | 0.366 | 0.648 | 0.209 | 0.491 |
| messidor_features | 1 | SU3 | p25 | 237 | 0.844 | 0.803 | 0.366 | 0.648 | 0.209 | 0.491 |
| messidor_features | 1 | SU3 | p50 | 160 | 0.944 | 0.932 | 0.466 | 0.669 | 0.388 | 0.543 |
| messidor_features | 1 | SU3 | p75 | 82 | 0.971 | 0.967 | 0.479 | 0.678 | 0.404 | 0.555 |
| messidor_features | 42 | B0 | baseline | 147 | 0.524 | 0.564 | 0.538 | 0.693 | 0.486 | 0.590 |
| messidor_features | 42 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| messidor_features | 42 | SU1 | 1.5 | 205 | 0.997 | 0.995 | 0.374 | 0.645 | 0.232 | 0.491 |
| messidor_features | 42 | SU1 | 2.0 | 226 | 0.971 | 0.970 | 0.366 | 0.648 | 0.209 | 0.491 |
| messidor_features | 42 | SU1 | 2.5 | 222 | 0.876 | 0.906 | 0.436 | 0.661 | 0.344 | 0.526 |
| messidor_features | 42 | SU1 | 3.0 | 249 | 0.868 | 0.848 | 0.399 | 0.656 | 0.276 | 0.509 |
| messidor_features | 42 | SU2 | median_parent_le5 | 248 | 0.955 | 0.941 | 0.456 | 0.667 | 0.374 | 0.538 |
| messidor_features | 42 | SU3 | p25 | 232 | 0.810 | 0.848 | 0.460 | 0.672 | 0.376 | 0.543 |
| messidor_features | 42 | SU3 | p50 | 174 | 0.907 | 0.918 | 0.384 | 0.648 | 0.254 | 0.497 |
| messidor_features | 42 | SU3 | p75 | 61 | 0.997 | 0.984 | 0.480 | 0.669 | 0.412 | 0.549 |
| spambase | 0 | B0 | baseline | 532 | 0.697 | 0.749 | 0.696 | 0.659 | 0.706 | 0.700 |
| spambase | 0 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| spambase | 0 | SU1 | 1.5 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| spambase | 0 | SU1 | 2.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| spambase | 0 | SU1 | 2.5 | 560 | 0.925 | 0.949 | 0.491 | 0.371 | 0.474 | 0.520 |
| spambase | 0 | SU1 | 3.0 | 666 | 0.911 | 0.917 | 0.476 | 0.319 | 0.439 | 0.524 |
| spambase | 0 | SU2 | median_parent_le5 | 690 | 0.948 | 0.943 | 0.488 | 0.238 | 0.374 | 0.611 |
| spambase | 0 | SU3 | p25 | 547 | 0.902 | 0.921 | 0.487 | 0.393 | 0.480 | 0.504 |
| spambase | 0 | SU3 | p50 | 407 | 0.938 | 0.956 | 0.420 | 0.144 | 0.285 | 0.551 |
| spambase | 0 | SU3 | p75 | 268 | 0.965 | 0.988 | 0.469 | 0.458 | 0.479 | 0.469 |
| spambase | 1 | B0 | baseline | 525 | 0.715 | 0.764 | 0.429 | 0.565 | 0.404 | 0.462 |
| spambase | 1 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| spambase | 1 | SU1 | 1.5 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| spambase | 1 | SU1 | 2.0 | 555 | 0.972 | 0.975 | 0.483 | 0.546 | 0.486 | 0.491 |
| spambase | 1 | SU1 | 2.5 | 579 | 0.927 | 0.943 | 0.479 | 0.578 | 0.470 | 0.498 |
| spambase | 1 | SU1 | 3.0 | 645 | 0.901 | 0.915 | 0.490 | 0.565 | 0.490 | 0.501 |
| spambase | 1 | SU2 | median_parent_le5 | 642 | 0.950 | 0.946 | 0.467 | 0.541 | 0.468 | 0.478 |
| spambase | 1 | SU3 | p25 | 530 | 0.895 | 0.916 | 0.420 | 0.561 | 0.393 | 0.454 |
| spambase | 1 | SU3 | p50 | 394 | 0.947 | 0.957 | 0.354 | 0.541 | 0.294 | 0.408 |
| spambase | 1 | SU3 | p75 | 263 | 0.979 | 0.985 | 0.381 | 0.570 | 0.322 | 0.438 |
| spambase | 42 | B0 | baseline | 551 | 0.712 | 0.779 | 0.479 | 0.541 | 0.483 | 0.486 |
| spambase | 42 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| spambase | 42 | SU1 | 1.5 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| spambase | 42 | SU1 | 2.0 | 555 | 0.980 | 0.981 | 0.496 | 0.574 | 0.495 | 0.508 |
| spambase | 42 | SU1 | 2.5 | 629 | 0.935 | 0.934 | 0.511 | 0.591 | 0.509 | 0.524 |
| spambase | 42 | SU1 | 3.0 | 595 | 0.916 | 0.928 | 0.460 | 0.569 | 0.448 | 0.482 |
| spambase | 42 | SU2 | median_parent_le5 | 643 | 0.960 | 0.961 | 0.526 | 0.574 | 0.534 | 0.531 |
| spambase | 42 | SU3 | p25 | 546 | 0.890 | 0.926 | 0.532 | 0.581 | 0.539 | 0.537 |
| spambase | 42 | SU3 | p50 | 417 | 0.925 | 0.962 | 0.467 | 0.537 | 0.468 | 0.476 |
| spambase | 42 | SU3 | p75 | 272 | 0.977 | 0.986 | 0.433 | 0.528 | 0.426 | 0.449 |
| tic_tac_toe | 0 | B0 | baseline | 124 | 0.349 | 0.674 | 0.258 | 0.515 | 0.000 | 0.347 |
| tic_tac_toe | 0 | SU1 | 1.0 | 177 | 1.000 | 0.998 | 0.258 | 0.515 | 0.000 | 0.347 |
| tic_tac_toe | 0 | SU1 | 1.5 | 198 | 0.987 | 0.982 | 0.258 | 0.515 | 0.000 | 0.347 |
| tic_tac_toe | 0 | SU1 | 2.0 | 204 | 0.862 | 0.904 | 0.258 | 0.515 | 0.000 | 0.347 |
| tic_tac_toe | 0 | SU1 | 2.5 | 209 | 0.750 | 0.868 | 0.258 | 0.515 | 0.000 | 0.347 |
| tic_tac_toe | 0 | SU1 | 3.0 | 205 | 0.703 | 0.858 | 0.258 | 0.515 | 0.000 | 0.347 |
| tic_tac_toe | 0 | SU2 | median_parent_le5 | 206 | 0.957 | 0.968 | 0.258 | 0.515 | 0.000 | 0.347 |
| tic_tac_toe | 0 | SU3 | p25 | 115 | 0.866 | 0.929 | 0.258 | 0.515 | 0.000 | 0.347 |
| tic_tac_toe | 0 | SU3 | p50 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| tic_tac_toe | 0 | SU3 | p75 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| tic_tac_toe | 1 | B0 | baseline | 126 | 0.267 | 0.674 | 0.258 | 0.515 | 0.000 | 0.347 |
| tic_tac_toe | 1 | SU1 | 1.0 | 176 | 1.000 | 1.000 | 0.258 | 0.515 | 0.000 | 0.347 |
| tic_tac_toe | 1 | SU1 | 1.5 | 196 | 0.978 | 0.982 | 0.258 | 0.515 | 0.000 | 0.347 |
| tic_tac_toe | 1 | SU1 | 2.0 | 207 | 0.849 | 0.922 | 0.258 | 0.515 | 0.000 | 0.347 |
| tic_tac_toe | 1 | SU1 | 2.5 | 201 | 0.677 | 0.874 | 0.258 | 0.515 | 0.000 | 0.347 |
| tic_tac_toe | 1 | SU1 | 3.0 | 210 | 0.625 | 0.858 | 0.258 | 0.515 | 0.000 | 0.347 |
| tic_tac_toe | 1 | SU2 | median_parent_le5 | 198 | 0.966 | 0.977 | 0.258 | 0.515 | 0.000 | 0.347 |
| tic_tac_toe | 1 | SU3 | p25 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| tic_tac_toe | 1 | SU3 | p50 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| tic_tac_toe | 1 | SU3 | p75 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| tic_tac_toe | 42 | B0 | baseline | 132 | 0.397 | 0.683 | 0.258 | 0.515 | 0.000 | 0.347 |
| tic_tac_toe | 42 | SU1 | 1.0 | 192 | 1.000 | 1.000 | 0.258 | 0.515 | 0.000 | 0.347 |
| tic_tac_toe | 42 | SU1 | 1.5 | 205 | 0.996 | 0.975 | 0.258 | 0.515 | 0.000 | 0.347 |
| tic_tac_toe | 42 | SU1 | 2.0 | 194 | 0.862 | 0.920 | 0.258 | 0.515 | 0.000 | 0.347 |
| tic_tac_toe | 42 | SU1 | 2.5 | 205 | 0.746 | 0.879 | 0.258 | 0.515 | 0.000 | 0.347 |
| tic_tac_toe | 42 | SU1 | 3.0 | 195 | 0.690 | 0.829 | 0.258 | 0.515 | 0.000 | 0.347 |
| tic_tac_toe | 42 | SU2 | median_parent_le5 | 202 | 0.978 | 0.989 | 0.258 | 0.515 | 0.000 | 0.347 |
| tic_tac_toe | 42 | SU3 | p25 | 126 | 0.853 | 0.906 | 0.258 | 0.515 | 0.000 | 0.347 |
| tic_tac_toe | 42 | SU3 | p50 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| tic_tac_toe | 42 | SU3 | p75 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| vehicle0 | 0 | B0 | baseline | 59 | 0.712 | 0.905 | 0.629 | 0.368 | 0.481 | 0.811 |
| vehicle0 | 0 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| vehicle0 | 0 | SU1 | 1.5 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| vehicle0 | 0 | SU1 | 2.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| vehicle0 | 0 | SU1 | 2.5 | 69 | 0.906 | 0.971 | 0.433 | 0.000 | 0.000 | 0.764 |
| vehicle0 | 0 | SU1 | 3.0 | 68 | 0.906 | 0.971 | 0.499 | 0.125 | 0.258 | 0.780 |
| vehicle0 | 0 | SU2 | median_parent_le5 | 75 | 0.978 | 0.991 | 0.553 | 0.229 | 0.363 | 0.787 |
| vehicle0 | 0 | SU3 | p25 | 57 | 0.914 | 0.980 | 0.433 | 0.000 | 0.000 | 0.764 |
| vehicle0 | 0 | SU3 | p50 | 42 | 0.971 | 0.991 | 0.499 | 0.125 | 0.258 | 0.780 |
| vehicle0 | 0 | SU3 | p75 | 28 | 1.000 | 1.000 | 0.693 | 0.510 | 0.631 | 0.803 |
| vehicle0 | 1 | B0 | baseline | 51 | 0.748 | 0.912 | 0.724 | 0.537 | 0.606 | 0.850 |
| vehicle0 | 1 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| vehicle0 | 1 | SU1 | 1.5 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| vehicle0 | 1 | SU1 | 2.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| vehicle0 | 1 | SU1 | 2.5 | 70 | 0.906 | 0.965 | 0.682 | 0.462 | 0.548 | 0.835 |
| vehicle0 | 1 | SU1 | 3.0 | 70 | 0.921 | 0.971 | 0.636 | 0.378 | 0.483 | 0.819 |
| vehicle0 | 1 | SU2 | median_parent_le5 | 66 | 0.978 | 0.993 | 0.467 | 0.065 | 0.183 | 0.772 |
| vehicle0 | 1 | SU3 | p25 | 60 | 0.935 | 0.971 | 0.467 | 0.065 | 0.183 | 0.772 |
| vehicle0 | 1 | SU3 | p50 | 34 | 1.000 | 0.993 | 0.635 | 0.400 | 0.531 | 0.787 |
| vehicle0 | 1 | SU3 | p75 | 27 | 1.000 | 0.993 | 0.543 | 0.255 | 0.421 | 0.724 |
| vehicle0 | 42 | B0 | baseline | 59 | 0.784 | 0.918 | 0.678 | 0.465 | 0.568 | 0.819 |
| vehicle0 | 42 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| vehicle0 | 42 | SU1 | 1.5 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| vehicle0 | 42 | SU1 | 2.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| vehicle0 | 42 | SU1 | 2.5 | 57 | 0.906 | 0.982 | 0.433 | 0.000 | 0.000 | 0.764 |
| vehicle0 | 42 | SU1 | 3.0 | 58 | 0.906 | 0.971 | 0.467 | 0.065 | 0.183 | 0.772 |
| vehicle0 | 42 | SU2 | median_parent_le5 | 76 | 0.978 | 0.993 | 0.499 | 0.125 | 0.258 | 0.780 |
| vehicle0 | 42 | SU3 | p25 | 46 | 0.942 | 0.991 | 0.428 | 0.000 | 0.000 | 0.748 |
| vehicle0 | 42 | SU3 | p50 | 23 | 0.986 | 0.998 | 0.544 | 0.238 | 0.393 | 0.748 |
| vehicle0 | 42 | SU3 | p75 | 21 | 1.000 | 1.000 | 0.509 | 0.258 | 0.448 | 0.638 |
| wholesale_customers | 0 | B0 | baseline | 51 | 0.148 | 0.733 | 0.422 | 0.176 | 0.386 | 0.525 |
| wholesale_customers | 0 | SU1 | 1.0 | 1 | 1.000 | 1.000 | nan | nan | nan | nan |
| wholesale_customers | 0 | SU1 | 1.5 | 35 | 1.000 | 1.000 | 0.639 | 0.400 | 0.552 | 0.797 |
| wholesale_customers | 0 | SU1 | 2.0 | 50 | 1.000 | 0.982 | 0.560 | 0.389 | 0.610 | 0.627 |
| wholesale_customers | 0 | SU1 | 2.5 | 49 | 0.889 | 0.964 | 0.370 | 0.327 | 0.455 | 0.373 |
| wholesale_customers | 0 | SU1 | 3.0 | 51 | 0.907 | 0.941 | 0.317 | 0.375 | 0.386 | 0.322 |
| wholesale_customers | 0 | SU2 | median_parent_le5 | 78 | 0.907 | 0.928 | 0.366 | 0.302 | 0.446 | 0.373 |
| wholesale_customers | 0 | SU3 | p25 | 63 | 0.778 | 0.864 | 0.264 | 0.338 | 0.312 | 0.271 |
| wholesale_customers | 0 | SU3 | p50 | 39 | 0.926 | 0.959 | 0.385 | 0.333 | 0.473 | 0.390 |
| wholesale_customers | 0 | SU3 | p75 | 14 | 0.981 | 0.995 | 0.476 | 0.111 | 0.273 | 0.729 |
| wholesale_customers | 1 | B0 | baseline | 45 | 0.130 | 0.670 | 0.507 | 0.231 | 0.452 | 0.661 |
| wholesale_customers | 1 | SU1 | 1.0 | 2 | 1.000 | 1.000 | nan | nan | nan | nan |
| wholesale_customers | 1 | SU1 | 1.5 | 32 | 1.000 | 1.000 | 0.508 | 0.267 | 0.500 | 0.627 |
| wholesale_customers | 1 | SU1 | 2.0 | 64 | 0.981 | 0.968 | 0.452 | 0.229 | 0.461 | 0.542 |
| wholesale_customers | 1 | SU1 | 2.5 | 67 | 0.889 | 0.955 | 0.417 | 0.298 | 0.502 | 0.441 |
| wholesale_customers | 1 | SU1 | 3.0 | 70 | 0.833 | 0.928 | 0.306 | 0.200 | 0.364 | 0.322 |
| wholesale_customers | 1 | SU2 | median_parent_le5 | 78 | 0.889 | 0.932 | 0.288 | 0.276 | 0.369 | 0.288 |
| wholesale_customers | 1 | SU3 | p25 | 69 | 0.685 | 0.869 | 0.244 | 0.333 | 0.289 | 0.254 |
| wholesale_customers | 1 | SU3 | p50 | 44 | 0.907 | 0.937 | 0.424 | 0.327 | 0.522 | 0.441 |
| wholesale_customers | 1 | SU3 | p75 | 29 | 0.981 | 0.982 | 0.391 | 0.071 | 0.246 | 0.559 |
| wholesale_customers | 42 | B0 | baseline | 43 | 0.185 | 0.647 | 0.558 | 0.364 | 0.583 | 0.644 |
| wholesale_customers | 42 | SU1 | 1.0 | 2 | 1.000 | 1.000 | nan | nan | nan | nan |
| wholesale_customers | 42 | SU1 | 1.5 | 43 | 1.000 | 1.000 | 0.432 | 0.182 | 0.393 | 0.542 |
| wholesale_customers | 42 | SU1 | 2.0 | 41 | 0.981 | 0.986 | 0.357 | 0.150 | 0.342 | 0.424 |
| wholesale_customers | 42 | SU1 | 2.5 | 52 | 0.907 | 0.946 | 0.433 | 0.279 | 0.484 | 0.475 |
| wholesale_customers | 42 | SU1 | 3.0 | 57 | 0.907 | 0.919 | 0.304 | 0.281 | 0.377 | 0.305 |
| wholesale_customers | 42 | SU2 | median_parent_le5 | 74 | 1.000 | 0.950 | 0.192 | 0.343 | 0.146 | 0.220 |
| wholesale_customers | 42 | SU3 | p25 | 64 | 0.759 | 0.819 | 0.157 | 0.314 | 0.000 | 0.186 |
| wholesale_customers | 42 | SU3 | p50 | 38 | 0.907 | 0.955 | 0.223 | 0.328 | 0.242 | 0.237 |
| wholesale_customers | 42 | SU3 | p75 | 15 | 0.981 | 0.977 | 0.494 | 0.125 | 0.279 | 0.763 |

## Outputs

- `split_utility_results.csv`
- `split_utility_summary.json`

