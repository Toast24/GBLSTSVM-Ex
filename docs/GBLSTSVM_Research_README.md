# GBLSTSVM Extension Research Log and Diagnostic README

## Project Overview

This repository investigates extensions to **Granular Ball Least Squares
Twin Support Vector Machine (GBLSTSVM)**.

The core question:

> Can we improve GBLSTSVM by making granular-ball construction and
> weighting mechanisms more reliable, especially under class imbalance,
> sparse minority support, and unstable topology estimates?

The existing pipeline consists of:

1.  Granular ball generation
2.  Ball filtering (`del_ball`)
3.  Ball-level weighting
4.  Twin SVM classification

The research focused on identifying exactly where performance failures
originate before proposing modifications.

------------------------------------------------------------------------

# Existing Baseline Components

## Granular Balls

The current granulation process:

-   Start with one ball containing all samples.
-   Recursively split using KMeans(k=2).
-   Split criterion:
    -   low purity
    -   sufficient samples

Each ball contains:

-   center
-   radius
-   purity
-   class label
-   samples

Important observation:

The current system strongly optimizes purity, often producing highly
pure balls.

At rho=0.95:

-   most final balls are almost pure
-   minority samples can become isolated small balls

------------------------------------------------------------------------

# CA (Class-Aware Weighting)

## Motivation

CA attempts to improve GBLSTSVM by considering local class competition.

A granular ball should receive a lower or higher weight depending on how
classes interact locally.

CA is primarily based on local neighbourhood/class information.

Intuition:

-   A ball surrounded by competing classes is important.
-   A ball in a pure homogeneous region is less informative.

CA captures:

> "How much does this ball participate in class discrimination?"

------------------------------------------------------------------------

# Topo (Topology-Based Weighting)

## Motivation

CA only looks at class-aware local structure.

Topo introduces graph-based information.

For each granular ball:

-   Construct relationships between balls.
-   Measure opposite-class support.
-   Estimate whether a ball lies near another class.

Important quantity:

## R_opp

Opposite-class reciprocal/topological support.

High R_opp:

-   strong opposite-class neighbourhood
-   reliable boundary information

Low/zero R_opp:

-   isolated ball
-   unreliable topology

Topo attempts to answer:

> "Is this ball genuinely supported by the surrounding data topology?"

------------------------------------------------------------------------

# Major Diagnostics Performed

## 1. Topo Failure Analysis

51 screening runs were analysed.

Findings:

Topo failures are not caused by:

-   radius alone
-   Gaussian overlap
-   nested balls

Failures occur when:

-   opposite-class support is sparse
-   R_opp becomes unstable
-   topology overreacts to isolated balls

Strong signals:

-   high fraction of R_opp=0 balls
-   large Topo-CA weight divergence

------------------------------------------------------------------------

# 2. Gaussian Membership Experiments

Tested Gaussian ball quality weighting:

## Formulations

### Compactness

mu = exp(-r\^2)

### Purity Compactness

mu = purity \* exp(-r\^2)

### Density Compactness

mu = density \* exp(-r\^2)

Purpose:

Determine whether geometric ball quality provides information
independent from CA/Topo.

Result:

-   Gaussian membership is almost independent from CA.
-   It provides geometric information.
-   It is not a replacement for Topo.

Failure:

Performance gains were inconsistent.

------------------------------------------------------------------------

# 3. Gaussian Overlap Diagnostic

Tested Bhattacharyya coefficient overlap between classes inside
heterogeneous balls.

Goal:

Determine whether failures come from mixed Gaussian distributions.

Result:

No.

The problem is not class overlap.

The problem is:

> Minority points are being isolated during granulation.

------------------------------------------------------------------------

# 4. Minority Isolation Analysis

Large-scale replay:

-   17 datasets
-   seeds 42/0/1
-   13,654 splits

Finding:

KMeans splitting often creates:

-   tiny minority child balls
-   singleton children

These are later removed by `del_ball`.

Evidence:

-   25% of splits isolate minority mass
-   80% of isolated children become singletons
-   minority survival drops significantly

------------------------------------------------------------------------

# 5. Singleton Repair Analysis

Question:

Can we identify good singleton minority points using only parent-local
information?

Answer:

No.

Parent-local features:

-   hd1
-   hd2
-   opposite neighbours
-   purity gain

cannot distinguish:

-   legitimate minority boundary points
-   true outliers

Maximum AUROC:

\~0.58

Need external support information.

------------------------------------------------------------------------

# 6. Split Utility Analysis

Final important discovery:

The issue is not simply singleton removal.

A geometric split quality measure works.

Best statistic:

## Center Radius Separation

distance between child centers divided by combined radii.

Result:

AUROC = 0.864

for separating:

-   harmful minority peel splits
-   useful retained splits

Parent size also predicts failure:

AUROC = 0.93

Small parents are dangerous:

Harmful splits:

median parent size = 3

Useful splits:

median parent size = 16

------------------------------------------------------------------------

# Current Research Hypothesis

The failure mode:

## Current GBLSTSVM creates unreliable granular boundaries.

Pipeline:

1.  Purity-driven KMeans split
2.  Minority samples get isolated
3.  Small balls are removed
4.  Topology loses support information
5.  Weighting becomes unstable

The main problem is not classifier formulation.

It is granulation quality.

------------------------------------------------------------------------

# Proposed Research Direction

## Adaptive Boundary-Aware Granulation

Do not redesign GBLSTSVM.

Modify only the split decision.

Before accepting a split:

Calculate:

-   center-radius separation
-   child compactness
-   parent size
-   minority concentration

If a split creates:

-   highly separated tiny minority balls
-   unstable boundary fragments

then:

-   prevent the split
-   merge children
-   or apply controlled resplitting

------------------------------------------------------------------------

# Experimental Roadmap

## Phase 1

Implement read-only split gate.

No classifier changes.

Evaluate:

-   minority survival
-   ball purity
-   number of balls
-   topology stability

------------------------------------------------------------------------

## Phase 2

Integrate into GBLSTSVM.

Compare:

Baseline GBLSTSVM

vs

CA-GBLSTSVM

vs

Topo-GBLSTSVM

vs

Adaptive Granulation + Topo

------------------------------------------------------------------------

## Phase 3

Statistical evaluation:

Datasets:

-   balanced datasets
-   imbalanced datasets
-   noisy datasets

Metrics:

-   Macro F1
-   Minority F1
-   G-mean
-   balanced accuracy

------------------------------------------------------------------------

# Current Best Understanding

The strongest evidence supports:

-   Do not add more weighting functions.
-   Do not add Gaussian membership alone.
-   Do not repair singleton balls after deletion.
-   Improve the granulation process itself.

The research contribution should be:

> A topology-aware adaptive granular ball construction mechanism that
> prevents unreliable minority fragmentation while preserving
> informative boundary balls.
