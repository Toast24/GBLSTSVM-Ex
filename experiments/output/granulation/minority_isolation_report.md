# Minority isolation granulation diagnostic

Read-only replay of frozen recursive KMeans(k=2) splits (rho=0.95, min_samples_per_ball=1, del_ball num_data=2).
No granulation, Topo, CW, or GBLSTSVM code modified.

Hypothesis: minority isolation → small-ball formation → del_ball/sample loss → sparse GB representation → poor classification.

Runs: **51** | splits: **13654**

## Mechanism tests (pre-specified)

| Test | Result |
|---|---|
| Minority vs majority survival (mean gap min−maj) | -0.1669 (frac runs min<maj=0.922) |
| Isolating splits whose minority child is later fully del_ball'd | 0.804 |
| Mean isolating-split rate | 0.252 |
| Mean preserving-split rate | 0.493 |
| hd1_norm isolating vs preserving (mean) | 1.918 vs 1.037 |
| local opp-support (2·hd1) isolating vs preserving | 1.589 vs 15.984 |
| KMeans purity gain isolating vs preserving | 0.429 vs 0.163 |
| Isolating splits that make a minority singleton | 0.801 |
| r(isolating split rate, minority survival) | -0.521 |

## Correlation with Topo Delta Macro F1 (Pearson r)

| Signal | r |
|---|---:|
| minority−majority survival | 0.223 |
| frac isolating splits | 0.050 |
| frac minority in small balls | -0.048 |
| minority survival | 0.164 |
| mean minority hd1_norm | -0.189 |
| mean local opp support (2·hd1) | -0.020 |
| mean n_opp (all balls) | 0.249 |
| mean n_opp (minority-label balls) | 0.217 |
| frac R_opp=0 | -0.111 |
| n deleted balls | 0.146 |
| n balls after del_ball | 0.229 |

## Informative vs adversarial means (R_opp category)

| Category | n | min survival | maj survival | frac isolating | frac min in small | mean n_opp min balls | Delta Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| adversarial_r_opp | 19 | 0.535 | 0.711 | 0.262 | 0.819 | 3.182 | -0.092 |
| informative_r_opp | 29 | 0.571 | 0.749 | 0.272 | 0.781 | 3.354 | 0.059 |
| redundant_r_opp | 3 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |

## Verdict

PARTIAL: the granulation chain is reproducible (isolation -> singleton/small ball -> del_ball minority loss -> sparse heterogeneous support). The last hop to poor classification is NOT supported: isolating-split rate does not correlate with Topo Delta Macro F1 and is essentially the same in adversarial vs informative R_opp runs. Do not modify the splitter for classifier gain yet. The repair below is justified only as a future granulation experiment whose success metric should be minority survival / small-ball rate, not Topo F1.

## Proposed splitter change (not implemented)

Smallest change, still KMeans(k=2), still rho=0.95, no downstream edits: after `split_2balls` assignment, repair only isolating children. If a child is small (n<=5) or singleton, carries the parent's local minority, and those points have heterogeneous support in the parent (n_opp_within_2*hd1 >= 2), reassign them to the sibling cluster. Leave unrepaired: (i) n_opp_within_2*hd1 == 1 outliers with large hd1_norm, (ii) mixed preserving splits. If repair would empty a child, skip that repair and keep the frozen KMeans partition. This keeps legitimate boundary minority inside a ball that survives del_ball, without changing KMeans(k=2), purity threshold, or GBLSTSVM. Implementation is deferred until a granulation-only replay is approved; do not screen GBLSTSVM yet.

## Outputs

- `minority_isolation_runs.csv`
- `minority_isolation_splits.csv`
- `minority_isolation_leaves.csv`
- `minority_isolation_samples.csv`
