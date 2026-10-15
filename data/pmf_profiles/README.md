# PMF profile data

Each motif contributes five umbrella-sampling PMF replicates to the manuscript
statistics. In the standard case these are `{motif}-1.xvg` through
`{motif}-5.xvg`. They provide the mean ± SD curves in Fig. 3 and Supplementary
Fig. S5 and the `replicate_SD` values in Table S3 (`data/reference_120.csv`).
Figure generation selects five files explicitly through
`replicate_series(..., count=5)` in
`scripts/figures/plot_data_analysis.py`.

A small number of additional run records are retained in this directory
(an extra replicate for a few motifs, and a few extended simulations). These
files do not enter the manuscript statistics. For hH037, the missing third
replicate is replaced by replicate 6; the sampling-quality script documents
and applies the same five-replicate convention.
