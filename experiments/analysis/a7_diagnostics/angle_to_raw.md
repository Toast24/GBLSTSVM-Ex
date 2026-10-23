# angle_to_raw.md

## Theta summary (degrees, vs RAW)

- **L1_POPULATION**: theta1 mean=37.78, median=29.54; theta2 mean=37.05, median=32.68
- **FULL_SCATTER**: theta1 mean=23.14, median=14.55; theta2 mean=23.22, median=16.40
- **DIAG_SCATTER**: theta1 mean=33.98, median=29.15; theta2 mean=36.13, median=30.85
- **ISOTROPIC_SCATTER**: theta1 mean=52.49, median=51.86; theta2 mean=51.89, median=51.91
- **RIDGE_TRACE_MATCH**: theta1 mean=52.49, median=51.86; theta2 mean=51.89, median=51.91
- **RIDGE_VAL_TUNED**: theta1 mean=41.71, median=37.43; theta2 mean=42.04, median=40.87

## Norm ratios vs RAW
- **L1_POPULATION**: ||w1||/||w1_RAW|| mean=14.200, ||w2||/||w2_RAW|| mean=16.908
- **FULL_SCATTER**: ||w1||/||w1_RAW|| mean=6.788, ||w2||/||w2_RAW|| mean=6.986
- **DIAG_SCATTER**: ||w1||/||w1_RAW|| mean=5.590, ||w2||/||w2_RAW|| mean=6.505
- **ISOTROPIC_SCATTER**: ||w1||/||w1_RAW|| mean=2.683, ||w2||/||w2_RAW|| mean=4.775
- **RIDGE_TRACE_MATCH**: ||w1||/||w1_RAW|| mean=2.683, ||w2||/||w2_RAW|| mean=4.775
- **RIDGE_VAL_TUNED**: ||w1||/||w1_RAW|| mean=7.148, ||w2||/||w2_RAW|| mean=13.930

## |log(w1/w2)| from params manifest norms
- **L1_POPULATION**: mean=0.8281
- **FULL_SCATTER**: mean=1.1323
- **DIAG_SCATTER**: mean=1.1401
- **ISOTROPIC_SCATTER**: mean=1.2672
- **RIDGE_TRACE_MATCH**: mean=1.2672
- **RIDGE_VAL_TUNED**: mean=0.8086
- **RAW_LSTSVM**: mean=1.1945

## Pairwise theta contrasts (dataset mean, MODE B)
- MODE_A_SHARED_C theta1 FULL_SCATTER-L1_POPULATION: mean=-21.79, median=-11.12, p=1.526e-05, BCa=[-34.72,-13.38]
- MODE_A_SHARED_C theta1 FULL_SCATTER-DIAG_SCATTER: mean=-13.74, median=-4.56, p=0.001068, BCa=[-24.41,-6.68]
- MODE_A_SHARED_C theta1 FULL_SCATTER-ISOTROPIC_SCATTER: mean=-34.53, median=-36.28, p=1.526e-05, BCa=[-42.69,-25.61]
- MODE_B_PER_ARM_C theta1 FULL_SCATTER-L1_POPULATION: mean=-7.49, median=-5.65, p=0.1901, BCa=[-20.40,-0.39]
- MODE_B_PER_ARM_C theta1 FULL_SCATTER-DIAG_SCATTER: mean=-7.94, median=-0.79, p=0.1454, BCa=[-17.22,-2.35]
- MODE_B_PER_ARM_C theta1 FULL_SCATTER-ISOTROPIC_SCATTER: mean=-24.16, median=-24.60, p=0.0003815, BCa=[-33.58,-14.81]