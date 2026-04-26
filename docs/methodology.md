# Methodology

## Overview

This document describes the computation pipeline used to generate all per-image gap analysis results in this repository. Every metric is deterministic and reproducible from the publicly available Kodak PNG files using Python 3, NumPy, and Pillow.

## Source Data

All 24 images from the Kodak Lossless True Color Image Suite (PCD0992) in their standard 768×512 base-resolution PNG format, sourced from [r0k.us/graphics/kodak/](http://r0k.us/graphics/kodak/). Each image is loaded as a raw 8-bit RGB pixel array with no preprocessing.

## Pipeline

### 1. PCA Baseline Replication

For each image, the RGB pixel data is reshaped into an (N×3) matrix where N = width × height. The following are computed directly from this matrix:

- **Pairwise Pearson correlations:** R-G, R-B, G-B, and average absolute |r|
- **3×3 covariance matrix** of the three channels
- **Eigendecomposition:** eigenvalues, eigenvectors, variance explained per component
- **Derived metrics:** condition number (λ₁/λ₃), eigenvalue ratios (λ₁/λ₂, λ₂/λ₃), blue channel independence (% of blue variance in PC2+PC3), PC1 dominant channel
- **Dimensionality tier:** classified by PC1 variance explained using fixed thresholds from Baetzel (2026a)

These values replicate the baselines from the parent paper and are carried forward into each per-image JSON to ensure exact correspondence between baseline and gap measurements.

### 2. BT.601 YCbCr Transform

The ITU-R BT.601 coefficient matrix is applied to the full RGB pixel array in floating-point precision:

```
Y  =  0.299R    + 0.587G    + 0.114B
Cb = −0.168736R − 0.331264G + 0.5B      + 128
Cr =  0.5R      − 0.418688G − 0.081312B + 128
```

No rounding or clipping is applied prior to correlation measurement. The +128 offset on Cb and Cr centers the chrominance channels but does not affect correlation computation.

### 3. YCbCr Residual Correlations

Pairwise Pearson correlation coefficients are computed between all three channel pairs in YCbCr space:

- **Y–Cb**, **Y–Cr**, **Cb–Cr**
- **Average absolute residual |r|**: mean of |Y–Cb|, |Y–Cr|, |Cb–Cr|

Because the KLT achieves zero residual correlation by construction, the post-BT.601 correlation directly defines the decorrelation gap.

### 4. Decorrelation Efficiency

```
Efficiency = (RGB avg |r| − YCbCr avg |r|) / RGB avg |r| × 100
```

This measures the fraction of original inter-channel correlation removed by the fixed transform, expressed as a percentage of the KLT optimum.

### 5. 4:2:0 Chroma Subsampling Simulation

The subsampling pipeline:

1. **Convert** RGB → YCbCr using BT.601 coefficients (as above)
1. **Downsample** Cb and Cr by 2× in both dimensions using box averaging (2×2 mean)
1. **Upsample** Cb and Cr back to full resolution using nearest-neighbor interpolation
1. **Inverse transform** YCbCr → RGB using BT.601 inverse coefficients
1. **Compute PSNR** between original and reconstructed RGB:
- Overall PSNR across all three channels
- Per-channel PSNR (R, G, B individually)

PSNR is computed as:

```
PSNR = 10 × log₁₀(255² / MSE)
```

where MSE is the mean squared error between original and reconstructed pixel values.

## Output Format

Each image produces one JSON file containing:

- RGB channel statistics (mean, std, min, max)
- RGB pairwise correlations and average |r|
- Full PCA decomposition (covariance matrix, eigenvalues, eigenvectors, variance explained)
- Dimensionality tier and derived metrics
- YCbCr channel statistics after BT.601
- YCbCr residual correlations (Y–Cb, Y–Cr, Cb–Cr) and average |r|
- Decorrelation gap and efficiency
- 4:2:0 PSNR (overall and per-channel)

One suite summary JSON aggregates means, ranges, and per-tier breakdowns across all 24 images.

## Dependencies

|Package |Role                                             |
|--------|-------------------------------------------------|
|Python 3|Runtime                                          |
|NumPy   |Correlation, covariance, eigendecomposition, PSNR|
|Pillow  |Image loading                                    |

No proprietary tools, no trained models, no parameters to tune. Every value is deterministic from the source PNGs.
