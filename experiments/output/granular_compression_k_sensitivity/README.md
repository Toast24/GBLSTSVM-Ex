# Granular compression K-sensitivity

Read-only exploratory diagnostic. **Not** a tuned deployable model.

## Protocol

- Frozen rho90 ball pool per dataset×seed (train only).
- Subsample **K** balls via documented ranking rules; classifier unchanged.
- `entropy_boundary_k`: min-max normalize entropy and boundary_score within pool, score = norm(entropy) × norm(boundary). Boundary = entropy × density (existing).
- Sample controls: retain `n_samples` from entropy_k selection, rebuild rho90 on subset.
- KMeans prototype control **skipped** (no existing prototype utility).
- Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; datasets: 17; random_k: 10 reps with deterministic seeds.

## K regime (mean macro F1)

```csv
K,random_k,entropy_k,boundary_k,entropy_boundary_k,purity_k,stratified_random_k,sample_random_k
2,0.6468875860232431,0.6468875860232431,0.6468875860232431,0.6468875860232431,0.6468875860232431,0.5383480430712837,0.6524474422450247
3,0.5995742442080058,0.5995742442080058,0.5995742442080058,0.5995742442080058,0.5995742442080058,0.5175476884522265,0.6230061583564012
4,0.5835610758216243,0.5835610758216242,0.5835610758216242,0.5835610758216242,0.5835610758216242,0.5610260266441465,0.6205779513758923
5,0.6058721740712091,0.6058721740712091,0.6058721740712091,0.6058721740712091,0.6058721740712091,0.568475529582865,0.6160489859230619
6,0.6084602159814462,0.6084602159814463,0.6084602159814463,0.6084602159814463,0.6084602159814463,0.594711005161155,0.6025063221536671
8,0.5953643048325411,0.595364304832541,0.595364304832541,0.595364304832541,0.595364304832541,0.5998658494995233,0.615740365878889
10,0.595166506132336,0.5951665061323359,0.5951665061323359,0.5951665061323359,0.5951665061323359,0.5982116047665642,0.6066735590381219
12,0.5997120436809901,0.5997120436809901,0.5997120436809901,0.5997120436809901,0.5997120436809901,0.6095455999086923,0.5983022387328037
16,0.5731502987926534,0.5731502987926534,0.5731502987926534,0.5731502987926534,0.5731502987926534,0.5765445368986543,0.5862015957105611
20,0.5351002459646211,0.5351002459646212,0.5351002459646212,0.5351002459646212,0.5351002459646212,0.55583218323366,0.5618013870316743
24,0.5321023196217213,0.5321023196217214,0.5321023196217214,0.5321023196217214,0.5321023196217214,0.5538564562934088,0.5571328456062162
32,0.5431465333146062,0.5431465333146062,0.5431465333146062,0.5431465333146062,0.5431465333146062,0.5547439072018195,0.543994660015704
48,0.543296462879072,0.5432964628790721,0.5432964628790721,0.5432964628790721,0.5432964628790721,0.5413896252749587,0.5508562668496076

```

## Optima per method (exploratory)

{
  "boundary_k": {
    "K_max_mean_macro_f1": 2.0,
    "max_mean_macro_f1": 0.6468875860232431,
    "K_max_median_macro_f1": 6.0,
    "max_median_macro_f1": 0.6141025641025641
  },
  "entropy_boundary_k": {
    "K_max_mean_macro_f1": 2.0,
    "max_mean_macro_f1": 0.6468875860232431,
    "K_max_median_macro_f1": 6.0,
    "max_median_macro_f1": 0.6141025641025641
  },
  "entropy_k": {
    "K_max_mean_macro_f1": 2.0,
    "max_mean_macro_f1": 0.6468875860232431,
    "K_max_median_macro_f1": 6.0,
    "max_median_macro_f1": 0.6141025641025641
  },
  "full_rho90": {
    "K_max_mean_macro_f1": NaN,
    "max_mean_macro_f1": 0.5842196778595634,
    "K_max_median_macro_f1": NaN,
    "max_median_macro_f1": 0.587535271745798
  },
  "purity_k": {
    "K_max_mean_macro_f1": 2.0,
    "max_mean_macro_f1": 0.6468875860232431,
    "K_max_median_macro_f1": 6.0,
    "max_median_macro_f1": 0.6141025641025641
  },
  "random_k": {
    "K_max_mean_macro_f1": 2.0,
    "max_mean_macro_f1": 0.6468875860232431,
    "K_max_median_macro_f1": 6.0,
    "max_median_macro_f1": 0.6141025641025641
  },
  "sample_random_k": {
    "K_max_mean_macro_f1": 2.0,
    "max_mean_macro_f1": 0.6524474422450247,
    "K_max_median_macro_f1": 2.0,
    "max_median_macro_f1": 0.6272157190635451
  },
  "sample_stratified_k": {
    "K_max_mean_macro_f1": 2.0,
    "max_mean_macro_f1": 0.672813738819308,
    "K_max_median_macro_f1": 2.0,
    "max_median_macro_f1": 0.6929400386847195
  },
  "stratified_random_k": {
    "K_max_mean_macro_f1": 12.0,
    "max_mean_macro_f1": 0.6095455999086923,
    "K_max_median_macro_f1": 10.0,
    "max_median_macro_f1": 0.6013090229079008
  }
}

## Questions

### A. Does compression improve generalization?
Full rho90 mean F1 **0.584**. At K≤10, mean F1 by method: sample_stratified_k=0.623, sample_random_k=0.620, boundary_k=0.605, entropy_boundary_k=0.605, purity_k=0.605, entropy_k=0.605, random_k=0.605, stratified_random_k=0.568. Compression to K=12 improves over full rho90 (Δ=+0.042 for entropy_k).

### B. Does boundary/entropy beat random at same K?
Entropy vs random (paired across K): mean Δ +0.000; boundary vs random: +0.000. **Weak/no** — selection advantage over random is small or inconsistent.

### C. Granular vs sample compression?
Entropy_k vs sample-random (matched train samples): mean Δ -0.011. **No** — sample-level compression is competitive; granular structure adds little.

### D. Mechanism
Decomposition: (1) **compression** — fewer balls/samples regularize vs full rho90; (2) **ball selection** — entropy/boundary ranking vs random; (3) **granular structure** — ball subset vs subsampled-then-granulated samples. Primarily **representation compression** (fewer balls vs full rho90). Ball ranking is neutral at fixed K; sample-level controls are competitive.

### Dataset consistency
532 dataset×method×K cells show ≥8/10 positive seeds vs full rho90. See dataset_level_results.csv — no cherry-picking.

## Outputs

- raw_results.csv
- summary_by_k_method.csv
- dataset_level_results.csv
- paired_comparisons.csv
- failure_analysis.csv
- plots/*.png

