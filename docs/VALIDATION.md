# GBB validation status

## 1. Synthetic recovery infrastructure

GBB includes a configurable mechanistic synthetic-data generator and a
parameter-recovery evaluation framework.

Synthetic datasets can contain known node-level ground truth for quantities
including:

- CfC effective timescale (`tau`)
- CfC intrinsic drive
- stimulus gain

Ground-truth parameters can be assigned using versioned ROI-specific JSON
profiles and are exported both as canonical node vectors and NIfTI maps.

## 2. Recovery evaluator self-test

The evaluator was tested using a deliberately perfect recovery condition:
the known synthetic ground-truth vectors were copied to the filenames normally
used for learned GBB parameter maps.

Expected result: exact recovery.

| Parameter | Pearson r | Spearman rho | MAE | RMSE | Slope | Intercept | Lin CCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| CfC effective timescale | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| CfC intrinsic drive | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 |

This is a software-validation test of the recovery pipeline. It demonstrates
that a perfectly recovered parameter vector produces the expected recovery
statistics.

**It is not evidence that a trained GBB model can recover these parameters
from synthetic fMRI data.**

## 3. End-to-end inverse-model recovery

Status: **pending**.

The next validation experiment trains GBB on synthetic CBV/BOLD generated
from known parameter maps and compares the subsequently learned maps with
the hidden ground truth.

The planned validation chain is:

synthetic ground truth
→ synthetic CBV/BOLD
→ GBB training
→ learned parameter maps
→ recovery metrics

Successful end-to-end recovery, rather than the evaluator self-test above,
is required before interpreting CfC parameters as identifiable model
quantities.
