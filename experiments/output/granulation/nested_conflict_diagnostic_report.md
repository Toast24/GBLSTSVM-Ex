# Nested-Ball Conflict Diagnostic (Mechanism 2)

Read-only inspection after granulation, **before** `del_ball`.
Purity=0.95, min_samples_per_ball=1, min_ball_count=2 (used only to flag would-remove).

## Per-run summary

| dataset | seed | balls | nested | hom nested | het conflict | % balls in conflict | del_ball touch | sparse graph | known Topo fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| diabetes | 42 | 296 | 6 | 3 | 3 | 2.0 | 100% of conflicts | False | False |
| diabetes | 0 | 290 | 6 | 4 | 2 | 1.4 | 100% of conflicts | False | False |
| diabetes | 1 | 300 | 5 | 2 | 3 | 2.0 | 100% of conflicts | False | False |
| ecoli2 | 42 | 31 | 0 | 0 | 0 | 0.0 | 0% of conflicts | False | True |
| ecoli2 | 0 | 26 | 0 | 0 | 0 | 0.0 | 0% of conflicts | False | True |
| ecoli2 | 1 | 28 | 0 | 0 | 0 | 0.0 | 0% of conflicts | False | True |
| vehicle0 | 42 | 126 | 4 | 3 | 1 | 1.6 | 100% of conflicts | False | False |
| vehicle0 | 0 | 142 | 1 | 1 | 0 | 0.0 | 0% of conflicts | False | False |
| vehicle0 | 1 | 126 | 1 | 1 | 0 | 0.0 | 0% of conflicts | False | True |

## Heterogeneous conflict details (by run)

### diabetes seed=42
- balls (14,143): labels (+1,-1), sizes (3,1), purities (1.000,1.000), radii (29.184,0.000), dist=23.700, either_del_ball=True
- balls (20,234): labels (-1,+1), sizes (11,1), purities (1.000,1.000), radii (15.508,0.000), dist=13.699, either_del_ball=True
- balls (165,168): labels (+1,-1), sizes (9,1), purities (1.000,1.000), radii (19.191,0.000), dist=15.812, either_del_ball=True

### diabetes seed=0
- balls (48,267): labels (+1,-1), sizes (2,1), purities (1.000,1.000), radii (25.327,0.000), dist=24.430, either_del_ball=True
- balls (238,276): labels (-1,+1), sizes (6,1), purities (1.000,1.000), radii (12.781,0.000), dist=12.570, either_del_ball=True

### diabetes seed=1
- balls (151,286): labels (-1,+1), sizes (18,1), purities (1.000,1.000), radii (22.225,0.000), dist=16.235, either_del_ball=True
- balls (186,188): labels (-1,+1), sizes (7,1), purities (1.000,1.000), radii (23.101,0.000), dist=17.756, either_del_ball=True
- balls (223,240): labels (-1,+1), sizes (4,1), purities (1.000,1.000), radii (9.552,0.000), dist=9.421, either_del_ball=True

### ecoli2 seed=42
- No heterogeneous nested conflicts.

### ecoli2 seed=0
- No heterogeneous nested conflicts.

### ecoli2 seed=1
- No heterogeneous nested conflicts.

### vehicle0 seed=42
- balls (12,25): labels (-1,+1), sizes (7,1), purities (1.000,1.000), radii (25.030,0.000), dist=23.131, either_del_ball=True

### vehicle0 seed=0
- No heterogeneous nested conflicts.

### vehicle0 seed=1
- No heterogeneous nested conflicts.

## Concentration analysis

Fraction of heterogeneous conflict pairs involving the listed ball property:

- Either small ball (n≤5): 100.0% (sparse-graph runs: 0.0%, dense runs: 100.0%)
- Either impure (purity<1): 0.0% (sparse: 0.0%, dense: 0.0%)
- Either minority-class ball: 100.0% (known Topo failure runs: 0.0%, other runs: 100.0%)
- Either would be removed by del_ball: 100.0%

### Sparse Topo failure runs

- **ecoli2 s42**: balls=31, k=5, het_conflicts=0, mean_n_opp=2.48, frac_R_opp=0=0.35
- **ecoli2 s0**: balls=26, k=5, het_conflicts=0, mean_n_opp=2.50, frac_R_opp=0=0.27
- **ecoli2 s1**: balls=28, k=5, het_conflicts=0, mean_n_opp=2.11, frac_R_opp=0=0.43
- **vehicle0 s1**: balls=126, k=7, het_conflicts=0, mean_n_opp=3.12, frac_R_opp=0=0.10

## Outputs

- Pair CSV: `nested_conflict_pairs.csv`
- Summary JSON: `nested_conflict_diagnostic_summary.json`
