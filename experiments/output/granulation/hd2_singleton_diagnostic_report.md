# hd2 Singleton Diagnostic (Mechanism 1)

Read-only replay of frozen recursive KMeans(k=2) splits.
Purity=0.95, min_samples_per_ball=1.
`del_ball` is **not** applied in this trace.

## Per-run summary

| dataset | seed | singletons | outliers | legitimate | undefined hd2 | samples affected |
|---|---:|---:|---:|---:|---:|---:|
| diabetes | 42 | 211 | 32 | 17 | 162 | 211 |
| diabetes | 0 | 206 | 26 | 14 | 166 | 206 |
| diabetes | 1 | 214 | 31 | 18 | 165 | 214 |
| ecoli2 | 42 | 18 | 2 | 0 | 16 | 18 |
| ecoli2 | 0 | 11 | 2 | 0 | 9 | 11 |
| ecoli2 | 1 | 11 | 4 | 1 | 6 | 11 |
| vehicle0 | 42 | 67 | 15 | 5 | 47 | 67 |
| vehicle0 | 0 | 83 | 20 | 9 | 54 | 83 |
| vehicle0 | 1 | 75 | 15 | 9 | 51 | 75 |

## Class distribution (all runs aggregated)

- All singleton children: {'+1': 434, '-1': 462}
- Predicted outliers: {'+1': 75, '-1': 72}
- Legitimate singletons: {'+1': 33, '-1': 40}

## Edge-case verification

### no_second_heterogeneous
- Count: 676
- All classified UNDEFINED_HD2: True
- Example: dataset=diabetes, seed=42, sample_index=329, parent_num=7, flags=no_second_heterogeneous

### duplicate_hd1_distance
- Count: 0

### singleton_parent
- Count: 0

### two_sample_parent
- Count: 484
- Example: dataset=diabetes, seed=42, sample_index=400, parent_num=2, flags=two_sample_parent|no_second_heterogeneous

## Outputs

- CSV: `hd2_singleton_diagnostic.csv`
- Summary JSON: `hd2_singleton_diagnostic_summary.json`
