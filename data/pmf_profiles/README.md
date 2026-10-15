# PMF profile data

Each motif is represented by five replicate umbrella-sampling PMF curves,
`{motif}-1.xvg` through `{motif}-5.xvg`. These five replicates are the sole
input to the mean ± SD curves in Fig. 3 and Supplementary Figs. S2 and S5, and
to the `replicate_SD` column of Table S3 (`data/reference_120.csv`). The
selection is explicit in `scripts/figures/plot_data_analysis.py`
(`replicate_series(..., count=5)`), so the same five files are always used.

A small number of additional run records are retained in this directory
(an extra replicate for a few motifs, and a few extended simulations). These
files are not part of the five-replicate set and do not enter any reported
statistic. Where an individual replicate file is missing, the plotting and
export scripts document the substitution at the point of use.
