# GBLSTSVM research reproduction and extensions

Clean reproduction of **Granular Ball Least Squares Twin SVM** (GBLSTSVM), plus controlled extensions for class weighting, ball reliability, and fuzzy ball-level membership mechanisms (CBCM, Topo, and related variants).

**Reference paper:** Tanveer, M., Sharma, R. K., Quadir, A., & Sajid, M. (2025). *Enhancing robustness and efficiency of least square twin SVM via granular computing.* Pattern Recognition, 112021. [https://doi.org/10.1016/j.patcog.2025.112021](https://doi.org/10.1016/j.patcog.2025.112021)

Session write-ups with results and inference live in [`../Sessions/`](../Sessions/) (especially [Session 3 — Topo and SupportAware-Topo](../Sessions/Session_3_Topo_and_SupportAware_Topo.md)).

---

## Repository layout

```text
GBLSTSVM/
├── src/                    # Clean implementations (use these)
│   ├── gblstsvm.py         # Frozen baseline
│   ├── cw_gblstsvm.py      # Class-weighted
│   ├── r_gblstsvm.py       # Reliability-weighted
│   ├── cr_gblstsvm.py      # CW × reliability
│   ├── cbcm_gblstsvm.py    # CA / CB / CC (CBCM)
│   ├── topology_gblstsvm.py # Topo + SupportAware-Topo
│   └── ...
├── tests/                  # Unit tests per mechanism
├── experiments/            # Screening, diagnostics, Phase 8 ablations
│   ├── dataset_registry.py # Auto-discovers local + UCI datasets
│   └── output/             # CSV/JSON/MD reports from runs
├── docs/                   # Mechanism docs and phase result summaries
├── Data/Data/              # Local headerless CSV benchmarks (4 files)
├── datasets/uci/           # Downloaded UCI datasets (optional bulk)
├── GBLSTSVM.py             # Original reference code (do not modify)
├── main.py                 # Original paper demo entry point
├── environment.yml         # Conda env (recommended)
└── requirements.txt        # Pip fallback
```

---

## Environment setup

**Requirements:** Python **3.11**, NumPy, SciPy, pandas, scikit-learn, matplotlib, `ucimlrepo` (for UCI downloads).

### Option A — Conda (recommended)

From the `GBLSTSVM` directory:

```bash
cd GBLSTSVM
conda env create -f environment.yml
conda activate GBenv
python -m unittest discover -s tests -v
```

### Option B — pip / venv

```bash
cd GBLSTSVM
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python -m unittest discover -s tests -v
```

### Verify the install

```bash
# All tests
python -m unittest discover -s tests -v

# Topology only
python -m unittest tests.test_topology tests.test_support_aware_topo -v
```

All experiment scripts assume the **current working directory is `GBLSTSVM`** (they add the project root to `sys.path` automatically).

---

## Data setup

Experiments load datasets from two sources (see `experiments/dataset_registry.py`):

| Source | Location | Notes |
|--------|----------|-------|
| Local benchmarks | `Data/Data/*.csv` | Headerless CSVs: `breast_cancer`, `diabetes`, `ecoli2`, `vehicle0` |
| UCI downloads | `datasets/uci/<name>/data.csv` | Fetched via `download_benchmark.py` |

### Download UCI benchmark sets

```bash
cd GBLSTSVM
conda activate GBenv   # or activate your venv
python download_benchmark.py
```

This creates `datasets/uci/<dataset_name>/` with `data.csv`, `metadata.json`, and `variables.csv`, plus `datasets/uci/download_log.json`.

Screening scripts call `discover_available_datasets()` and skip datasets that are missing or fail to load.

---

## Frozen experimental protocol

Unless a script states otherwise, screening experiments use:

| Setting | Value |
|---------|-------|
| Split | Stratified **70% train / 15% val / 15% test** |
| Seeds | `{42, 0, 1}` for broad screens; `42` for cheap validation |
| `c1` / `c2` | `100.0` / `0.01` |
| Purity | `0.95` |
| `min_ball_count` | `2` |
| Preprocessing | **None** (no scaling) |
| Prediction rule | `reference` |
| Weight placement | `opposite_only` (for weighted variants) |

**No hyperparameter tuning after seeing test results.** Cheap screens are go/no-go gates; broad screens compare methods under the same splits.

---

## Model families

| Stage | Class | Weight formula | Status |
|-------|-------|----------------|--------|
| Baseline | `GBLSTSVM` | uniform | Frozen |
| Class weight | `CWGBLSTSVM` | κ(y) | Frozen |
| Reliability | `RGBLSTSVM` | a_i | Frozen |
| Combined | `CRGBLSTSVM` | κ(y)·a_i | Frozen |
| CBCM competition | `CAGBLSTSVM` | κ(y)·μ^A | Best fuzzy signal so far |
| CBCM connectivity | `CBGBLSTSVM` | κ(y)·μ^B | Failed / not promoted |
| CBCM combined | `CCGBLSTSVM` | κ(y)·μ^C | Failed / not promoted |
| BMRU / OBRU | `BMRUGBLSTSVM`, `OBRUGBLSTSVM` | margin / overlap | Failed |
| IPA | `IPAGBLSTSVM` | ambiguity / isolation | Failed |
| **Topo** | `TopoGBLSTSVM` | κ(y)·A·R_opp | Localized gains; not promoted to CV |
| **SupportAware-Topo** | `SupportAwareTopoGBLSTSVM` | κ(y)·A·R_opp^S | Partial fix for sparse-graph failures |

Mechanism docs: [`docs/`](docs/) (start with [`CLASS_WEIGHTING.md`](docs/CLASS_WEIGHTING.md), [`CBCM_GBLSTSVM.md`](docs/CBCM_GBLSTSVM.md), [`TOPOLOGY_GBLSTSVM.md`](docs/TOPOLOGY_GBLSTSVM.md)).

---

## How to run experiments

**Convention:** run from `GBLSTSVM`:

```bash
cd GBLSTSVM
conda activate GBenv
python experiments/<script>.py
```

Outputs land under `experiments/output/<topic>/` unless noted.

### Phase 1–5 — Baseline and CW / R / CR

| Script | Purpose | Output |
|--------|---------|--------|
| `experiments/compare_reference.py` | Compare clean vs reference implementation | console |
| `experiments/validate_baseline.py` | Baseline validation | `experiments/output/` |
| `experiments/validate_class_weighting.py` | CW screening | `experiments/output/` |
| `experiments/validate_reliability.py` | R screening | `experiments/output/` |
| `experiments/validate_cr_gblstsvm.py` | CR screening | `experiments/output/` |
| `experiments/analyze_ball_reliability.py` | Ball reliability diagnostics | `experiments/output/` |

Docs: [`docs/BASELINE_RECONCILIATION.md`](docs/BASELINE_RECONCILIATION.md), [`docs/RELIABILITY_WEIGHTING.md`](docs/RELIABILITY_WEIGHTING.md), [`docs/CR_GBLSTSVM.md`](docs/CR_GBLSTSVM.md).

### Phase 6–7 — Real datasets (breast cancer + multi-dataset)

| Script | Purpose | Output |
|--------|---------|--------|
| `experiments/validate_breast_cancer.py` | First real-dataset protocol | `experiments/output/breast_cancer_*` |
| `experiments/validate_multidataset.py` | diabetes, ecoli2, vehicle0 | `experiments/output/` |

Docs: [`docs/BREAST_CANCER_EXPERIMENT.md`](docs/BREAST_CANCER_EXPERIMENT.md), [`docs/PHASE7_MULTIDATASET_RESULTS.md`](docs/PHASE7_MULTIDATASET_RESULTS.md).

### Phase 8 — Ablations (scaling, granulation, multiseed)

| Script | Purpose |
|--------|---------|
| `experiments/phase8_scaling_ablation.py` | Preprocessing ablation |
| `experiments/phase8_granulation_sensitivity.py` | Purity / min_ball_count |
| `experiments/phase8_hyperparameter_grid.py` | C1/C2 grid (exploratory) |
| `experiments/phase8_multiseed.py` | Multi-seed stability |
| `experiments/phase8_mechanism_diagnostics.py` | Mechanism diagnostics |
| `experiments/phase8_final_protocol.py` | Combined best-protocol run |
| `experiments/phase8_plots.py` | Generate phase-8 plots |

Shared utilities: `experiments/phase8_common.py`. Summary: [`docs/PHASE8_RESULTS.md`](docs/PHASE8_RESULTS.md).

### BMRU and OBRU (failed extensions)

| Script | Purpose |
|--------|---------|
| `experiments/validate_bmru.py` | BMRU screening |
| `experiments/analyze_bmru_diagnostics.py` | BMRU diagnostics |
| `experiments/validate_overlap.py` | OBRU screening |
| `experiments/analyze_overlap.py` | OBRU diagnostics |

Docs: [`docs/BMRU_GBLSTSVM.md`](docs/BMRU_GBLSTSVM.md), [`docs/OVERLAP_GBLSTSVM.md`](docs/OVERLAP_GBLSTSVM.md).

### CBCM (competition–connectivity membership)

Run diagnostics **before** the 5-arm CV evaluation.

| Script | Purpose | Output |
|--------|---------|--------|
| `experiments/cbcm_diagnostics.py` | Pre-training gate | `experiments/output/cbcm/` |
| `experiments/analyze_cbcm_competition.py` | Competition forensics | `experiments/output/cbcm/` |
| `experiments/validate_cbcm.py` | 5-arm stratified CV | `experiments/output/cbcm/` |

Doc: [`docs/CBCM_GBLSTSVM.md`](docs/CBCM_GBLSTSVM.md).

### IPA (failed)

| Script | Purpose |
|--------|---------|
| `experiments/analyze_ipa.py` | Pre-training diagnostics |
| `experiments/validate_ipa.py` | Cheap CW / CA / IPA screen |

Doc: [`docs/IPA_GBLSTSVM.md`](docs/IPA_GBLSTSVM.md).

### Topo and SupportAware-Topo (latest)

Recommended order:

```bash
# 1. Unit tests
python -m unittest tests.test_topology tests.test_support_aware_topo -v

# 2. Pre-training diagnostics (diabetes, ecoli2, vehicle0)
python experiments/analyze_topology.py

# 3. Cheap 3-dataset screen (CW, CA, Topo)
python experiments/validate_topology.py

# 4. Broad screen — 17 datasets × 3 seeds (GBLSTSVM, CW, CA, Topo)
python experiments/topo_broad_screening.py

# 5. Mechanism analysis on completed Topo runs
python experiments/analyze_r_opp_mechanism.py

# 6. Ball-level forensics (6 curated success vs adversarial runs)
python experiments/analyze_r_opp_forensics.py

# 7. SupportAware-Topo screen (adds SupportAware-Topo; original Topo unchanged)
python experiments/topo_support_aware_screening.py
```

| Script | Output |
|--------|--------|
| `analyze_topology.py` | `experiments/output/topology/topology_diagnostics_report.md` |
| `validate_topology.py` | `experiments/output/topology/topology_validation_report.md` |
| `topo_broad_screening.py` | `topo_broad_screening_results.csv`, `_summary.json` |
| `analyze_r_opp_mechanism.py` | `r_opp_mechanism_summary.json` |
| `analyze_r_opp_forensics.py` | `r_opp_forensics_summary.json`, per-ball CSV |
| `topo_support_aware_screening.py` | `topo_support_aware_report.md`, screening CSVs |

Doc: [`docs/TOPOLOGY_GBLSTSVM.md`](docs/TOPOLOGY_GBLSTSVM.md). Session inference: [`../Sessions/Session_3_Topo_and_SupportAware_Topo.md`](../Sessions/Session_3_Topo_and_SupportAware_Topo.md).

---

## Quick reference — latest Topo results

Broad screen (51 runs = 17 datasets × 3 seeds), vs CA:

| Method | Macro F1 W/T/L | Mean Δ |
|--------|----------------|--------|
| Topo | 16 / 15 / 20 | −0.0008 |
| SupportAware-Topo | 15 / 20 / 16 | **+0.0031** |

SupportAware-Topo removes catastrophic failures on sparse graphs (ecoli2, ionosphere, vehicle0) but can erase Topo wins when `n_opp` is low everywhere (e.g. chronic_kidney_disease). **Neither variant is promoted to CV.**

---

## Original paper code (legacy)

The untouched reference implementation from the authors:

| File | Role |
|------|------|
| `main.py` | Demo runner |
| `GBLSTSVM.py` | Optimization solver |
| `gen_ball.py` | Granular ball generation |
| `addNoisy.py` | Label noise utility |

Example hyperparameters for `breast_cancer` in the original demo: `c1=1000`, `c2=1e-5`.

The clean stack in `src/` is what all research experiments use. Do not modify the legacy files when extending mechanisms.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: experiments` | Run scripts from `GBLSTSVM`, not the repo root |
| `ModuleNotFoundError: ucimlrepo` | `pip install ucimlrepo` or recreate the conda env |
| Screening finds fewer than 17 datasets | Run `python download_benchmark.py`; ensure `Data/Data/*.csv` exists |
| Tests fail after env change | `pip install -r requirements.txt` or `conda env update -f environment.yml` |

---

## Citation

If you use the original GBLSTSVM code, cite the Tanveer et al. (2025) paper linked above.
