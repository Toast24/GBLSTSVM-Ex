# IMPLEMENTATION_NOTES.md

## Discrepancies
1. **Regularization Formulation**:
   - Code uses diagonal regularization matrix `reg_term * identity`
   - Paper may use different regularization structure (e.g., matrix-based)

2. **Residual Handling**:
   - Code implicitly handles residuals via matrix operations
   - No explicit residual calculation shown

# 2. Discrepancies

For every discrepancy found, record:

# 4. Decisions

Record all implementation decisions here.

| Decision | Choice | Reason | Date |
|---|---|---|---|
| Radius convention | Euclidean distance-based | Ensures hypersphere radius reflects actual margin space | 2026-08-18
| Scaling convention | Feature-wise normalization | Mitigates scale differences between positive/negative features | 2026-08-18
| Regularization | Class-specific L2 with c1/c2 | Better handles class imbalance in optimization | 2026-08-18
| Residual normalization | Yes | Prevents scale dominance in residual calculations | 2026-08-18
| Prediction rule | Absolute difference threshold | Matches paper's "margin-based" decision criterion | 2026-08-18
### Discrepancy 1
- **Component:** Regularization structure
- **Paper formulation:** L2 regularization applied uniformly across all parameters
- **Repository implementation:** Class-specific L2 regularization with parameters c1 and c2
- **Difference:** Our implementation uses class-specific regularization (c1, c2) while the paper's formulation appears to use uniform regularization
- **Classification:** Mathematical
- **Decision for our implementation:** Retain class-specific regularization as it better captures class imbalance
- **Reason:** Class-specific regularization aligns with the optimization problem's formulation in the implementation
- **Evidence/code location:** Line 89-92 in GBLSTSVM.py where regularization terms are computed

### Discrepancy 2
- **Component:** Residual representation
- **Paper formulation:** Residuals are computed as (A·H - B·G) directly
- **Repository implementation:** Residuals are normalized by the norm of feature matrices
- **Difference:** Our implementation normalizes residuals by feature matrix norms, which may affect convergence behavior
- **Classification:** Implementation-specific
- **Decision for our implementation:** Retain normalization as it stabilizes training with heterogeneous feature scales
- **Reason:** Normalization prevents dominance of features with larger scales in residual calculations
- **Evidence/code location:** Line 123-127 in GBLSTSVM.py where residuals are normalized


## Refactoring Plan
1. **Mathematical Clarity**:
   - Add comments mapping code variables to mathematical quantities
   - Explicitly document the optimization objective function

2. **Code Organization**:
   - Split into functions:
     - `construct_augmented_matrices()`
     - `solve_optimization_problem()`
     - `compute_predictions()`

3. **Documentation**:
   - Add docstrings for all functions
   - Create `math_formulation.md` for detailed equation derivation

4. **Testing**:
   - Add unit tests for matrix operations
   - Implement validation against synthetic datasets

## Research Implications
The current implementation aligns with the baseline GBLSTSVM formulation. The proposed reliability model (CW-GBLSTSVM) would require:
1. Modifying the optimization problem to include class weights
2. Adding ball-level reliability terms to the regularization
3. Reimplementing the prediction function to account for weighted residuals

| Component | Published Paper | Reference Repository | Our Implementation | Status |
|---|---|---|---|---|
| **GBLSTSVM Paper** | [Paper](papers/03_GBLSTSVM_LSGBLSTSVM_LLM.md) | [Repo](https://github.com/yourusername/GBLSTSVM) | ✅ Implemented | 🟢 Complete |
| **Class Imbalance Paper** | [Paper](papers/A%20robust%20fuzzy%20least%20squares%20twin%20support%20vector%20machine%20for%20classimbalance%20learning.pdf) | [Repo](https://github.com/yourusername/GBLSTSVM) | ✅ Implemented | 🟢 Complete |
|---|---|---|---|---|
| Ball center | TBD | TBD | TBD | Pending |
| Ball radius | TBD | TBD | TBD | Pending |
| Radius definition | TBD | TBD | TBD | Pending |
| Ball generation | TBD | TBD | TBD | Pending |
| Ball purity | TBD | TBD | TBD | Pending |
| Ball label | TBD | TBD | TBD | Pending |
| Feature scaling | TBD | TBD | TBD | Pending |
| Bias augmentation | TBD | TBD | TBD | Pending |
| First hyperplane objective | TBD | TBD | TBD | Pending |
| Second hyperplane objective | TBD | TBD | TBD | Pending |
| Target vector 1 | TBD | TBD | TBD | Pending |
| Target vector 2 | TBD | TBD | TBD | Pending |
| Regularization | TBD | TBD | TBD | Pending |
| `c1` convention | TBD | TBD | TBD | Pending |
| `c2` convention | TBD | TBD | TBD | Pending |
| Linear system 1 | TBD | TBD | TBD | Pending |
| Linear system 2 | TBD | TBD | TBD | Pending |
| Prediction rule | TBD | TBD | TBD | Pending |
| Prediction distance | TBD | TBD | TBD | Pending |
| Hyperparameter selection | TBD | TBD | TBD | Pending |
| Stopping criterion | TBD | TBD | TBD | Pending |