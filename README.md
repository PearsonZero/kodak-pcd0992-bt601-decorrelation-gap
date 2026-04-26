# Per-Image Decorrelation Efficiency of the BT.601 Fixed Transform

**Measured Against KLT Ceiling Across the Kodak Lossless True Color Image Suite**

Baetzel, J. (2026b)

> **Part 2 of the Kodak Inter-Channel Redundancy Series** — measures BT.601 performance against the per-image KLT baselines established in [Statistical Characterization of Inter-Channel Redundancy Structure](https://github.com/PearsonZero/kodak-pcd0992-statistical-characterization).

-----

This repository contains the first per-image measurement of how much inter-channel decorrelation the standard JPEG color transform (BT.601 YCbCr) actually achieves — versus the theoretical optimum (KLT) — across all 24 images in the Kodak Lossless True Color Image Suite.

**Key result:** BT.601 removes only ~55% of available inter-channel correlation on average, leaving a large, structured residual — primarily between the two chroma channels (Cb–Cr), which retain a mean correlation of 0.616. This redundancy passes largely untouched through standard channel-independent encoder stages, constituting a measurable, persistent inefficiency in every JPEG-family pipeline since 1992.

## Key Findings

|Metric                           |Value                            |
|---------------------------------|---------------------------------|
|Mean decorrelation efficiency    |55.4% of KLT optimum             |
|Mean residual correlation (YCbCr)|0.372                            |
|Efficiency range across images   |31.5% – 82.3%                    |
| Cb–Cr residual mean \|r\| | 0.616 |
|Cb–Cr residual range             |0.299 – 0.934                    |
|4:2:0 PSNR range                 |39.91 – 47.91 dB (8.00 dB spread)|
|Cb–Cr is dominant residual in    |23 of 24 images                  |

**See the data:** [kodim01](PDF%20and%20JSON/kodim01_gap_analysis.pdf) (near suite mean — Cb–Cr: 0.660, PSNR: 45.07 dB) · [kodim14](PDF%20and%20JSON/kodim14_gap_analysis.pdf) (worst-case — Cb–Cr: 0.630, PSNR: 39.91 dB)

## What’s in This Repo

```
/
├── README.md
├── baetzel_2026_bt601_decorrelation_gap.pdf    ← Full paper
└── PDF and JSON/
    ├── suite_summary.json                       ← Aggregate metrics
    ├── kodim01_gap_analysis.json                ← Per-image data (×24)
    ├── kodim01_gap_analysis.pdf                 ← Per-image data sheet (×24)
    ├── ...
    ├── kodim24_gap_analysis.json
    └── kodim24_gap_analysis.pdf
```

Each per-image JSON contains the complete measurement set: RGB channel statistics, PCA decomposition (replicated from Baetzel 2026a), YCbCr residual correlations, decorrelation gap and efficiency, and 4:2:0 chroma subsampling PSNR (overall and per-channel). Every value in the paper traces to a specific field in a specific JSON.

## How to Reproduce

Everything is independently verifiable from publicly available data.

**Requirements:** 24 Kodak PNGs from [r0k.us/graphics/kodak/](http://r0k.us/graphics/kodak/), Python 3, NumPy, Pillow.

**Steps:**

1. Load each image as a raw 8-bit RGB pixel array
1. Compute pairwise Pearson correlations and eigendecomposition (replicates Paper 1)
1. Apply BT.601: Y = 0.299R + 0.587G + 0.114B; Cb = −0.169R − 0.331G + 0.5B + 128; Cr = 0.5R − 0.419G − 0.081B + 128
1. Compute pairwise Pearson correlations in YCbCr space
1. Simulate 4:2:0: downsample Cb/Cr by 2× (box average), upsample (nearest neighbor), apply BT.601 inverse, compute PSNR against original RGB

No proprietary tools, no trained models, no parameters to tune. Every value is deterministic from the source PNGs.

## Related Work

**Baseline paper:** Baetzel, J. (2026a). *Statistical Characterization of Inter-Channel Redundancy Structure in the Kodak Lossless True Color Image Suite.* Per-Image Principal Component Decomposition of PCD0992. Establishes the per-image KLT ceiling, dimensionality tiers, blue channel independence spectrum, and evidence for deliberate curation. Published separately; cited throughout this paper as the source of per-image PCA baselines.

**This paper** measures what BT.601 actually achieves against those ceilings and documents the structure of what it leaves behind.

## References

1. Wallace, G.K. “The JPEG still picture compression standard.” *IEEE Transactions on Consumer Electronics* 38(1), 1992.
1. ITU-R Recommendation BT.601, 1982.
1. Watanabe, S. “Karhunen–Loève Expansion and Factor Analysis.” pp. 635–660, 1965.
1. Baetzel, J. (2026a). “Statistical Characterization of Inter-Channel Redundancy Structure in the Kodak Lossless True Color Image Suite.”
1. Gershikov, E. and Porat, M. “Does decorrelation really improve color image compression?” *ICSTSC*, 2005.
1. Gershikov, E., Lavi-Burlak, E. and Porat, M. “Correlation-based approach to color image compression.” *Signal Processing: Image Communication* 22, pp. 719–733, 2007.

## Citation

```
Baetzel, J. (2026b). Per-Image Decorrelation Efficiency of the BT.601
Fixed Transform Measured Against KLT Ceiling Across the Kodak Lossless
True Color Image Suite. Residual Correlation Structure and Chroma
Subsampling Cost in PCD0992.
```

## License

Statistical analysis and data sheets by Jasmine Baetzel (2026). Benchmark images from the Kodak Lossless True Color Image Suite (PCD0992), released by Eastman Kodak Company for unrestricted usage.
