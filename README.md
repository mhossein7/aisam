# AISAM: AI scientist for automated microscopy

## Standard training data pipeline

```mermaid
graph TD
    root[root_folder]
    cfg[config.json<br/>circuit/label<br/>t_max<br/>sampling<br/>num_realizations<br/>num_cells=1000<br/>output_root optional<br/>random_seed optional]
    params[simulation_params.json<br/>circuit parameters<br/>CcaSR: alpha, k, n, tau_delay, h1, h2, c2, delta<br/>Inverter adds beta, k_tet, n_tet]
    noisy_opts[optional noisy settings<br/>noisy_sims<br/>temperatures]
    model_opts[optional model settings<br/>model type<br/>past/future windows<br/>feature/output species<br/>sample_interval_minutes]

    run[training.run_training_simulation]
    stims[standard 1000-cell stims<br/>1-900 random<br/>901-950 repetitive red-first<br/>951-1000 repetitive green-first]
    sim[standard GillesPy simulation<br/>1000 cells x num_realizations]
    saved[run folder<br/>{label}_{timestamp}<br/>simulation.pkl<br/>simulation_params.json<br/>stims.json<br/>config.json]

    noisy[optional noisy simulations<br/>sample noisy parameter dictionaries<br/>1000 cells each]
    noisy_saved[noisy/sim_i folders<br/>simulation.pkl<br/>simulation_params.json<br/>stims.json<br/>config.json]

    train[optional downstream training<br/>training.training]
    preprocess[forecaster preprocessing<br/>sample input/expression traces<br/>at sample_interval_minutes]
    model[model artifacts<br/>model.pkl<br/>config.json<br/>metrics]

    root --> cfg
    root --> params
    cfg --> run
    params --> run
    noisy_opts --> run
    run --> stims
    stims --> sim
    run --> sim
    sim --> saved
    noisy_opts --> noisy
    saved --> noisy
    noisy --> noisy_saved
    model_opts --> train
    saved --> train
    train --> preprocess
    preprocess --> model
```

## Codebase dependency graph

```mermaid
graph TD
    aisam[aisam package]
    model_pkg[aisam.model]
    utils_pkg[aisam.utils]
    comptools_pkg[aisam.comptools]

    training_py[aisam/utils/training.py]
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

    training_py --> sims_py
    training_py --> aux_py
    training_py --> json

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
    run_cli[training.run_training_simulation]

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
