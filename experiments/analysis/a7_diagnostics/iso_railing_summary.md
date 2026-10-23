# iso_railing_summary.md

## Railing rate (MODE_A_SHARED_C)
- **L1_POPULATION**: c1@min=0.265, c2@min=0.241 (n=170 runs per rule)
- **FULL_SCATTER**: c1@min=0.265, c2@min=0.241 (n=170 runs per rule)
- **DIAG_SCATTER**: c1@min=0.265, c2@min=0.241 (n=170 runs per rule)
- **ISOTROPIC_SCATTER**: c1@min=0.265, c2@min=0.241 (n=170 runs per rule)
- **RIDGE_TRACE_MATCH**: c1@min=0.265, c2@min=0.241 (n=170 runs per rule)
- **RIDGE_VAL_TUNED**: c1@min=0.265, c2@min=0.241 (n=170 runs per rule)

## Railing rate (MODE_B_PER_ARM_C)
- **L1_POPULATION**: c1@min=0.265, c2@min=0.241 (n=170 runs per rule)
- **FULL_SCATTER**: c1@min=0.171, c2@min=0.194 (n=170 runs per rule)
- **DIAG_SCATTER**: c1@min=0.135, c2@min=0.112 (n=170 runs per rule)
- **ISOTROPIC_SCATTER**: c1@min=0.165, c2@min=0.135 (n=170 runs per rule)
- **RIDGE_TRACE_MATCH**: c1@min=0.165, c2@min=0.135 (n=170 runs per rule)
- **RIDGE_VAL_TUNED**: c1@min=0.165, c2@min=0.135 (n=170 runs per rule)

- Dose-response Spearman (MODE_A_SHARED_C, RULE_REPO): rho=-0.1250, p=0.6326 (n=17 datasets)
- Dose-response Spearman (MODE_A_SHARED_C, RULE_REFERENCE): rho=-0.0049, p=0.9851 (n=17 datasets)
- Dose-response Spearman (MODE_B_PER_ARM_C, RULE_REPO): rho=-0.2451, p=0.343 (n=17 datasets)
- Dose-response Spearman (MODE_B_PER_ARM_C, RULE_REFERENCE): rho=0.0245, p=0.9256 (n=17 datasets)
- ISOTROPIC−L1 mean delta (MODE_A_SHARED_C, RULE_REPO): -0.1581
- ISOTROPIC−L1 mean delta (MODE_A_SHARED_C, RULE_REFERENCE): -0.0854
- ISOTROPIC−L1 mean delta (MODE_B_PER_ARM_C, RULE_REPO): -0.0275
- ISOTROPIC−L1 mean delta (MODE_B_PER_ARM_C, RULE_REFERENCE): -0.0822

Effective ridge column: `geom_trace_isotropic` from scatter_geometry (equals sum_j tr(S_j) for isotropic arm); per-class lambdas in ridge_matching.