# Section 6 Historical Civilizations Replication

This repository contains the public replication code for Section 6, "Application 2: Historical Civilizations," of *Unveiling hidden features of social evolution by inferring Langevin dynamics from data*.

The package focuses on the Polaris/Seshat Scale-Computation application. It prepares Natural Geographic Area (NGA) trajectories, fits the nonparametric Gaussian-process SDE implementation used for the section, and exports perturbation and irreversibility diagnostics.

## Repository Contents

- `src/process_polaris_data.py`: converts cleaned Polaris/Seshat data into the Section 6 analysis schema.
- `src/run_polaris_analysis.py`: trains the npSDE model and writes per-NGA diagnostics.
- `src/npsde_pyro.py`: Pyro implementation of the Yildiz-style npSDE estimator and transition diagnostics.
- `data/scv_clean_all_fixed.csv`: cleaned input table used by the preprocessing command.
- `data/scv_processed_for_npsde.csv`: prepared Section 6 input used by the analysis command.
- `scv_analysis_outputs/`: selected output summaries, metrics, and figures for inspecting the reported Section 6 cases.

Other exploratory analyses and generated model checkpoints are intentionally excluded from the public package.

## Setup

Create an environment with Python 3.11 or newer, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Data Schema

The analysis command expects a CSV with these columns:

```text
NGA, Year, Time, Scale, Computation
```

- `NGA`: Natural Geographic Area label.
- `Year`: historical year attached to the observation.
- `Time`: within-NGA time index in centuries, starting at 1.
- `Scale`: material magnitude index.
- `Computation`: information-processing and administrative-capacity index.

The current prepared file, `data/scv_processed_for_npsde.csv`, has 266 observations across 27 NGAs. The paper text describes approximately 250 observations and 27 NGAs; the small row-count difference reflects the cleaned replication snapshot in this repository after dropping rows without usable Scale/Computation values.

## Reproduction Commands

To rebuild the prepared analysis CSV from the cleaned input:

```bash
python -m src.process_polaris_data \
  --input data/scv_clean_all_fixed.csv \
  --output data/scv_processed_for_npsde.csv
```

To run the Section 6 npSDE analysis:

```bash
python -m src.run_polaris_analysis \
  --input data/scv_processed_for_npsde.csv \
  --output-dir scv_analysis_outputs
```

For a quick smoke run while checking installation:

```bash
python -m src.run_polaris_analysis \
  --train-steps 1 \
  --Nw 1 \
  --metrics-samples 2 \
  --plot-samples 1 \
  --ngas Latium \
  --output-dir tmp_smoke_outputs
```

Training and transition-density diagnostics are stochastic. Use `--seed` to make NumPy, PyTorch, and Pyro initialization reproducible for a given software environment.

## Outputs

Deterministic preprocessing outputs:

- `prepared_for_npsde.csv`: analysis-ready `Label, Time, x1, x2` table written inside the selected output directory.

Stochastic model outputs:

- `analysis_summary.json`: run metadata plus relative paths to generated NGA metrics and plots.
- `<NGA>_metrics.csv`: per-observation Scale, Computation, forward transition log density, backward transition log density, and log-ratio diagnostics.
- `<NGA>_aligned_plots.png`: diagnostic plot for selected NGAs.

Model checkpoint files (`*.pt`) and full model diagnostic PNGs are ignored by git by default because they are generated artifacts.

## Data and License Notes

The Polaris/Seshat data derive from the Seshat: Global History Databank. Confirm the redistribution terms for any public release or archival deposit before treating the data files as redistributable. See `LICENSE` for the current code-license decision status.
