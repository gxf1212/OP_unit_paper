# OP_unit_paper

Public result-level data, selected molecular-dynamics inputs and paper-specific
scripts for the manuscript *Machine Learning-Guided Prioritization of
Zwitterionic Polymer Motifs for Membrane Interaction Simulations*.

Result-level reproduction uses the processed files in `data/` to regenerate
reported statistics, tables, and figures. Model-level reproduction uses the
training scripts but additionally requires the specified external MolBlender
installation and its optional dependencies.

## Reproduction

Create an environment from `requirements.txt`, then run commands from the
repository root. The following commands regenerate the central LOGO figures
from the released processed data:

```bash
python scripts/figures/plot_logo_landscape.py \
  --output manuscript/Figures/Fig4_Screening_Landscape.png
python scripts/figures/plot_logo_performance.py
python scripts/figures/manuscript/plot_external_predictions_S6.py
python scripts/figures/manuscript/plot_S7_lead_sensitivity.py
```

The remaining manuscript and Supplementary Information figure scripts follow
the same convention and write to `manuscript/Figures/`. Figure 3 is generated
with `python scripts/figures/plot_data_analysis.py --pmf-selection fixed-rgroup`.
Temporary figures, caches, model checkpoints, trajectories, and SQLite
databases are not part of the release.

The released supplementary tables can be validated or regenerated without
modifying the source data:

```bash
mkdir -p outputs
python scripts/figures/manuscript/export_tableS3_reference_dataset.py \
  > outputs/TableS3_reference_dataset_120.csv
python scripts/figures/manuscript/export_tableS6_prospective_pmf.py
python scripts/figures/manuscript/export_tableS7_prospective_library.py
```

The LOGO scripts use five target classes (A/N, C, H, P and S) and
`random_state=42`. Candidate representation-model combinations were screened
within each LOGO training partition; the selected configuration was then
retrained and evaluated on the isolated target class by
`scripts/logo/retrain_logo_best_model.py`. The deployment analysis used a
five-fold split, retained 16 eligible finalists, evaluated all 6,868
two-to-five-member ensembles, and selected a five-member ensemble. The 16
single-model results were retained as separate comparators and are not part of
the 6,868 ensemble count.

The historical LOGO screening was run with MolBlender snapshot
`c0b58c6c1524f26917b4301d77b1ecfb589de878`. MolBlender itself is not bundled
in this paper repository. To inspect or rerun model-level code, provide a
compatible checkout through `MOLBLENDER_ROOT` (or install it on `PYTHONPATH`)
and install the model-specific optional dependencies. The processed screening,
external-test and out-of-fold outputs needed for result-level reproduction are
included under `data/`.

MD-input review uses the selected files in `md_inputs/` to inspect the
force-field, equilibration and umbrella-window settings. The release does not
contain the complete 120-motif run tree or production trajectories.

## Data

The 120-motif reference table, 32-candidate library, eight prospective PMF
results, LOGO screening output, external predictions, ensemble exports, and
lead-sensitivity export are under `data/`. The corresponding manuscript and SI
figure scripts are under `scripts/figures/`.

`data/logo_screening_output.csv` corresponds to Supplementary Data 1;
`data/reference_120.csv`, `data/prospective_pmf_8.csv`, and
`data/prospective_32.csv` correspond to Tables S3, S6, and S7, respectively.
`data/logo_metrics.csv` and `data/logo_external_predictions.csv` contain the
external LOGO metrics and per-motif predictions, whereas
`data/logo_cv_full_outer_summary.csv` contains the training-partition CV
values shown in Fig. 5a.

## Molecular-dynamics inputs

`md_inputs/mdp/` contains the PMF equilibration and production parameter files.
`md_inputs/example_C003/` contains one representative topology and equilibrated
system, while `md_inputs/forcefield/` contains the project-level topology files
used by that example. `md_inputs/scripts/` contains ligand parameterization
helpers and the distance utility. These files document the study-specific
settings and provide a representative reconstruction starting point. The
example is not directly runnable until the referenced `amber14sb_OL15.ff`
dependencies are restored. The complete historical cluster directory is not
reproduced; GROMACS (including WHAM for umbrella-sampling analysis), CHARMM-GUI
conversion and ACPYPE/Antechamber environments are also required.

## License

Code in `scripts/` is released under the MIT License (see `LICENSE`).
Data in `data/` and `md_inputs/` are released under the Creative Commons
Attribution 4.0 International License (CC BY 4.0; see `LICENSE-DATA`).
