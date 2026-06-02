# AISAM: AI scientist for automated microscopy

## Standard training data pipeline

```mermaid
graph TD
    root["root_folder"]
    cfg["config.json<br/>circuit/label<br/>t_max<br/>sampling<br/>num_realizations<br/>total_cell >= 200<br/>output_root optional<br/>random_seed optional"]
    params["simulation_params.json<br/>circuit parameters<br/>CcaSR: alpha, k, n, tau_delay, h1, h2, c2, delta<br/>Inverter adds beta, k_tet, n_tet"]
    noisy_opts["optional noisy settings<br/>noisy_sims<br/>temperatures"]
    model_opts["optional model settings<br/>model type<br/>past/future windows<br/>feature/output species<br/>sample_interval_minutes"]

    run["simulation.run_training_simulation"]
    stims["standard stims<br/>first total_cell - 100 random<br/>next 50 repetitive red-first<br/>last 50 repetitive green-first"]
    sim["standard GillesPy simulation<br/>total_cell x num_realizations"]
    out["training_data directory<br/>root_folder/training_data<br/>or assets/yymmdd/training_data<br/>when called with only config path"]
    saved["timestamped run folder<br/>simulation.parquet primary<br/>simulation.pkl legacy/debug<br/>simulation_params.json<br/>stims.json<br/>config.json"]

    noisy["optional noisy simulations<br/>sample noisy parameter dictionaries<br/>noisy_total_cells default 350<br/>250 random + 100 repetitive by default"]
    noisy_saved["noisy/sim_i folders<br/>simulation.parquet<br/>simulation.pkl legacy/debug<br/>simulation_params.json<br/>stims.json<br/>config.json"]

    train["optional downstream training<br/>forecaster_training.train_forecaster"]
    preprocess["forecaster preprocessing<br/>sample input/expression traces<br/>at sample_interval_minutes"]
    model["model artifacts in run folder<br/>models/model.pkl<br/>models/config.json<br/>metrics"]

    root --> cfg
    root --> params
    cfg --> run
    params --> run
    noisy_opts --> run
    run --> out
    run --> stims
    stims --> sim
    run --> sim
    out --> saved
    sim --> saved
    noisy_opts --> noisy
    saved --> noisy
    noisy --> noisy_saved
    model_opts --> train
    saved --> train
    train --> preprocess
    preprocess --> model
```

## Example CcaSR pipeline

For a full CcaSR standard simulation with optional noisy simulations and a regression forecaster, the user supplies a root folder containing:

- `config.json`: run settings such as `circuit`, `label`, `t_max`, `sampling`, `total_cell`, `num_realizations`, optional `noisy_sims`, optional `noisy_total_cells`, optional `temperatures`, and optional model settings.
- `simulation_params.json`: CcaSR parameters `alpha`, `k`, `n`, `tau_delay`, `h1`, `h2`, `c2`, and `delta`.

Minimal example `config.json`:

```json
{
  "circuit": "ccasr",
  "label": "ccasr",
  "t_max": 960,
  "sampling": 10,
  "total_cell": 1000,
  "num_realizations": 3,
  "root_folder": "/path/to/run_root",
  "noisy_sims": 2,
  "noisy_total_cells": 350,
  "temperatures": [0.05, 0.1],
  "sample_interval_minutes": 5,
  "model": {
    "type": "regressor",
    "past_feature_window": 20,
    "future_window": 1,
    "output_species": "F"
  }
}
```

Example `simulation_params.json`:

```json
{
  "alpha": 0.1,
  "k": 0.4851,
  "n": 3.6,
  "tau_delay": 12,
  "h1": 0.07100805,
  "h2": 0.0303,
  "c2": 0.0631,
  "delta": 0.01
}
```

Run the full pipeline:

```python
from aisam.utils.pipeline import training

result = training("/path/to/run_root")
```

The standard simulation is saved under `root_folder/training_data/{label}_{timestamp}/`. The primary data artifact is `simulation.parquet`, a long-form table with columns `cell_id`, `realization`, `species`, `stim`, `time`, and `value`. `simulation.pkl` is still written by default as a legacy/debug artifact. If `noisy_sims` is greater than zero, noisy simulations are saved under that run folder in `noisy/sim_i/`. Forecaster training reads the parquet file by default and saves model artifacts under the run folder in `models/`.

## Recipe-based experiment

An entire experiment can also be defined with `recipe.json` in the run root. If the same folder also contains `config.json` or `simulation_params.json`, those files override the recipe/default simulation values.

```json
{
  "mode": "full",
  "circuit": "ccasr",
  "simulation_model": "gillespy_tau_hybrid",
  "forecaster_model": {
    "type": "regressor",
    "past_feature_window": 20,
    "future_window": 1,
    "output_species": "F"
  },
  "include_model_training": true,
  "include_sanity_check": true,
  "include_periodic_stims_in_training": false,
  "include_periodic_stims_in_validation": true,
  "t_max": 960,
  "sampling": 10,
  "interval_rate": 5,
  "total_cell": 1000,
  "num_realizations": 3,
  "noisy_sims": 0
}
```

Run from the recipe folder:

```bash
aisam run
```

Or point AISAM to another recipe root:

```bash
aisam run --path /path/to/run_root
```

`mode` can be `simulation`, `training`, `full`, or `sanity`. `aisam simulate` remains available as a compatibility alias for `aisam run`. If only `recipe.json` exists, AISAM uses default circuit parameters stored in `aisam.model.defaults`.

## Codebase dependency graph

```mermaid
graph TD
    aisam[aisam package]
    model_pkg[aisam.model]
    utils_pkg[aisam.utils]
    comptools_pkg[aisam.comptools]

    simulation_py[aisam/utils/simulation.py]
    forecaster_py[aisam/utils/forecaster_training.py]
    pipeline_py[aisam/utils/pipeline.py]
    training_py[aisam/utils/training.py<br/>compatibility re-exports]
    sims_py[aisam/model/sims.py]
    models_py[aisam/model/models.py]
    aux_py[aisam/utils/aux.py]
    tf_py[aisam/comptools/transformer_forecaster.py]

    gillespy2[gillespy2]
    numpy[numpy]
    scipy[scipy.integrate]
    torch[torch]
    matplotlib[matplotlib]
    dill[dill]
    json[json]

    pipeline_py --> simulation_py
    pipeline_py --> forecaster_py
    training_py --> pipeline_py
    training_py --> simulation_py
    training_py --> forecaster_py
    simulation_py --> sims_py
    simulation_py --> aux_py
    simulation_py --> json
    forecaster_py --> simulation_py
    forecaster_py --> comptools_pkg

    aisam --> model_pkg
    aisam --> utils_pkg
    aisam --> comptools_pkg

    sims_py --> models_py
    sims_py --> aux_py
    sims_py --> gillespy2
    sims_py --> numpy
    sims_py --> dill
    sims_py --> json

    models_py --> gillespy2
    models_py --> numpy
    models_py --> scipy

    aux_py --> numpy
    aux_py --> matplotlib
    aux_py --> json

    tf_py --> torch
```

## Functional call graph (high-level)

```mermaid
graph TD
    run_cli[simulation.run_training_simulation]

    load_cfg[Load config JSON]
    load_params[aux.load_params]

    exp_init[sims.experiment.__init__]
    init_exp[sims.experiment.init_exp]
    run_train[sims.experiment.run_training_sim]
    save_outputs[Save simulation.pkl/config/stims/params]

    cell_ctor[Cell_sim]
    assign_params[Cell_sim.assign_parameters]
    assign_model[Cell_sim.assign_model]
    model_init[model.init_rxn]
    assign_features[Cell_sim.assign_features]

    run_multi[RXN.run_multi_rxn]
    update_rxn[RXN.update_rxn]
    gilles_run[gillespy2.Model.run]
    assign_run[Cell_sim.assign_run]

    run_cli --> load_cfg
    run_cli --> load_params
    run_cli --> exp_init
    run_cli --> init_exp
    run_cli --> run_train
    run_cli --> save_outputs

    init_exp --> cell_ctor
    init_exp --> assign_params
    init_exp --> assign_model
    init_exp --> model_init
    init_exp --> assign_features

    run_train --> run_multi
    run_multi --> update_rxn
    run_multi --> gilles_run
    run_train --> assign_run
```

## Model architecture (transformer forecaster)

```mermaid
graph TD
    X[Feature history X]
    U[Input history U]

    feat_embed[Linear feature embedding]
    inp_embed[Linear input embedding]
    pos[Learnable positional embedding]

    block1[TransformerBlock x N\nself-attn + cross-attn + FFN]
    last[Last token state]
    out_proj[Output linear projection]
    y[Predicted next feature vector]

    X --> feat_embed
    U --> inp_embed
    pos --> feat_embed
    pos --> inp_embed

    feat_embed --> block1
    inp_embed --> block1
    block1 --> last
    last --> out_proj
    out_proj --> y
```
