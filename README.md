# AISAM: AI Scientist for Automated Microscopy

AISAM runs recipe-driven simulation experiments for synthetic microscopy/circuit data and trains a forecaster model to predict future species values from past observations and stimulation history.

The main workflow is:

1. Write a `recipe.json`.
2. Run `aisam run --path /path/to/experiment`.
3. Inspect the saved simulation data and forecaster performance report.

## Quick Start

Create an experiment folder:

```text
experiments/ccasr_demo/
  recipe.json
```

Add a minimal full-pipeline `recipe.json`:

```json
{
  "mode": "full",
  "circuit": "ccasr",
  "solver": "gillespy_tau_hybrid",
  "forecaster_model": {
    "type": "regressor",
    "past_feature_window": 20,
    "future_window": 1,
    "output_species": "F",
    "validation_fraction": 0.2,
    "visualization": true
  },
  "t_max": 960,
  "sampling": 10,
  "interval_rate": 5,
  "total_cell": 1000,
  "num_realizations": 1,
  "include_periodic_stims_in_training": false,
  "include_periodic_stims_in_validation": true,
  "save_parquet": true,
  "save_pickle": false,
  "random_seed": 1
}
```

Run it:

```bash
aisam run --path experiments/ccasr_demo
```

AISAM will simulate cells, save the simulation table, train the forecaster, and evaluate it on held-out data.

## What You Need For A Full Pipeline

- A folder containing `recipe.json`.
- `mode: "full"` to run simulation and forecaster training together.
- A circuit choice such as `ccasr`, `inverter`, `CcaSR_noE`, `CcaSR_ODE`, `Inverter_noE`, or `ODE_Inverter`.
- A solver choice. Use `gillespy_tau_hybrid` for stochastic/GillesPy2 circuits and `deterministic` for ODE circuits.
- Simulation size settings: `t_max`, `sampling`, `interval_rate`, `total_cell`, and `num_realizations`.
- A `forecaster_model` block. The current standard model type is `regressor`.
- Optional circuit parameters. If omitted, AISAM uses defaults from `aisam.model.defaults`. To override them, add a `circuit_parameters` block in the recipe or put `simulation_params.json` next to the recipe.

`total_cell` must be at least `200`. By default, the stimulation panel includes random-stimulation cells plus repetitive stimulation cells for evaluation or training policies.

## Example With Parameter Overrides

```json
{
  "mode": "full",
  "circuit": "CcaSR_noE",
  "solver": "gillespy_tau_hybrid",
  "forecaster_model": {
    "type": "regressor",
    "past_feature_window": 20,
    "future_window": 1,
    "output_species": "F"
  },
  "circuit_parameters": {
    "alpha": 1,
    "k": 0.4851,
    "n": 3.6,
    "tau_delay": 12,
    "c2": 0.0631,
    "delta": 0.01
  },
  "t_max": 960,
  "sampling": 10,
  "interval_rate": 5,
  "total_cell": 1000,
  "num_realizations": 1
}
```

## Outputs

A full run creates a timestamped simulation folder under:

```text
experiments/ccasr_demo/training_data/{circuit}_{timestamp}/
```

Important files:

- `simulation.parquet`: the main simulation dataset.
- `simulation.pkl`: optional legacy/debug cell object output, controlled by `save_pickle`.
- `simulation_params.json`: parameter values used for the run.
- `stims.json`: stimulation sequence for each cell.
- `config.json`: run metadata and saved paths.
- `models/{model_label}_{timestamp}/model.pkl`: trained forecaster.
- `models/{model_label}_{timestamp}/performance.json`: train/evaluation metrics.
- `models/{model_label}_{timestamp}/config.json`: model metadata, policies, metrics, and data counts.
- `models/{model_label}_{timestamp}/figures/`: optional plots when `visualization` is enabled.

The parquet file is a long-form table with:

```text
cell_id, realization, species, stim, time, value
```

The performance report includes regression metrics such as `mse`, `rmse`, `mae`, and `r2` for training and evaluation.

## Run Only Part Of The Pipeline

Simulation only:

```json
{
  "mode": "simulation",
  "circuit": "ccasr",
  "solver": "gillespy_tau_hybrid",
  "t_max": 960,
  "sampling": 10,
  "interval_rate": 5,
  "total_cell": 1000,
  "num_realizations": 1
}
```

Training only from an existing simulation folder or parquet file:

```json
{
  "mode": "training",
  "simulation_path": "/path/to/training_data/ccasr_2026-06-05_12-00-00/simulation.parquet",
  "forecaster_model": {
    "type": "regressor",
    "past_feature_window": 20,
    "future_window": 1,
    "output_species": "F",
    "visualization": true
  }
}
```

Then run the same command:

```bash
aisam run --path /path/to/experiment
```

You can also train directly from saved simulation data:

```bash
aisam train --path /path/to/simulation.parquet --config regressor --visualization
```

Cross-test a forecaster by training on one simulation dataset and evaluating on
another:

```bash
aisam predict \
  --model regressor \
  --training-data /path/to/training_run/simulation.parquet \
  --test-data /path/to/test_run/simulation.parquet
```

Or evaluate an existing trained model on a new simulation dataset:

```bash
aisam predict \
  --trained-model /path/to/models/regressor_run/model.pkl \
  --test-data /path/to/test_run/simulation.parquet
```

Cross-testing outputs are saved under `models/{model}/cross_testing/{timestamp}/`.
They include `training_holdout_performance.json` when training-holdout data is
available, `test_performance.json`, a combined `performance.json`, and figures
for both `figures/training_holdout/` and `figures/test/`: `rmse_histogram.svg`,
`best_prediction.svg`, `worst_prediction.svg`, and `random_prediction.svg`.

## Useful Recipe Fields

- `mode`: `full`, `simulation`, `training`, or `sanity`.
- `circuit`: the circuit/model family to simulate.
- `solver`: simulation backend. Use `deterministic` for ODE circuits and `gillespy_tau_hybrid` for stochastic circuits.
- `forecaster_model`: model configuration for training.
- `interval_rate`: saved time step in minutes for downstream training data.
- `include_periodic_stims_in_training`: whether repetitive stimulation cells are part of training.
- `include_periodic_stims_in_validation`: whether repetitive stimulation cells are part of evaluation.
- `noisy_sims`, `noisy_total_cells`, `temperatures`: optional noisy parameter simulations.
- `save_parquet`, `save_pickle`: control simulation output formats.

## Programmatic Use

```python
from aisam.utils.pipeline import run_recipe

result = run_recipe("experiments/ccasr_demo")
print(result["simulation"]["simulation_path"])
print(result["model"]["performance_path"])
print(result["model"]["metrics"]["evaluation"])
```
