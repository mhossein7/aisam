import json
from datetime import datetime
from os import PathLike
from pathlib import Path

import dill
import numpy as np

from aisam.model import sims
from aisam.utils import aux


def training(config, model=None):
    """
    Run the standard simulation pipeline and optionally train a forecaster model.

    `config` can be a dictionary or a JSON file path. Expected keys:
    - circuit or label: circuit name, defaults to "CcaSR"
    - params or circuit_parameters: parameter dictionary
    - params_path: optional JSON path used when params are not provided directly
    - t_max: stimulation length; also added to params when missing
    - sampling: simulation sampling; also added to params when missing
    - num_realizations: number of trajectories per cell, defaults to 1
    - random_seed: optional seed for reproducible standard stim generation
    - output_root: optional override for the output directory
    - model: optional model config. Supported model type: "regressor"

    `model` can also be passed directly as a string or dictionary and overrides
    config["model"].

    Returns a dictionary of saved paths and run metadata.
    """
    config = _load_config(config)
    label = config.get("label", config.get("circuit", "CcaSR"))
    num_realizations = int(config.get("num_realizations", 1))
    num_cells = int(config.get("num_cells", 1000))

    if num_cells != 1000:
        raise ValueError("The standard training simulation currently expects num_cells=1000.")

    params = _load_params_from_config(config)
    if config.get("t_max", params.get("t_max")) is None:
        raise ValueError("Provide t_max in the config or parameter dictionary.")

    t_max = int(config.get("t_max", params.get("t_max")))
    sampling = int(config.get("sampling", params.get("sampling", 10)))
    params.setdefault("t_max", t_max)
    params.setdefault("sampling", sampling)

    random_seed = config.get("random_seed")
    if random_seed is not None:
        np.random.seed(int(random_seed))

    stims = build_standard_stims(t_max)
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_root = Path(config.get("output_root", default_training_data_root()))
    run_dir = output_root / f"{label}_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    xpt = sims.experiment(params, label)
    xpt.init_exp(num_cells)
    xpt.run_training_sim(stims, num_realizations)

    simulation_path = run_dir / "simulation.pkl"
    with open(simulation_path, "wb") as f:
        dill.dump(xpt.Cells, f)

    params_path = run_dir / "simulation_params.json"
    with open(params_path, "w") as f:
        json.dump(_json_safe(params), f, indent=2)

    stims_path = run_dir / "stims.json"
    with open(stims_path, "w") as f:
        json.dump(stims, f, indent=2)

    config_dump = {
        "created_at": run_stamp,
        "circuit": label,
        "circuit_parameters": _json_safe(params),
        "simulation_file": str(simulation_path),
        "simulation_params_file": str(params_path),
        "stims_file": str(stims_path),
        "simulated_cells": {
            "total_cells": num_cells,
            "random_stimulation_cells": [1, 900],
            "repetitive_stimulation_cells_red_first": [901, 950],
            "repetitive_stimulation_cells_green_first": [951, 1000],
            "num_realizations": num_realizations,
            "t_max": t_max,
            "sampling": sampling,
        },
        "user_config": _json_safe(config),
    }

    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dump, f, indent=2)

    result = {
        "run_dir": run_dir,
        "simulation_path": simulation_path,
        "params_path": params_path,
        "stims_path": stims_path,
        "config_path": config_path,
        "config": config_dump,
    }

    model_config = _resolve_model_config(model if model is not None else config.get("model"))
    if model_config is not None:
        model_result = _train_and_save_model(
            model_config=model_config,
            cells=xpt.Cells,
            data_config=config_dump,
            run_stamp=run_stamp,
        )
        result["model"] = model_result

    return result


def build_standard_stims(total_time):
    stims = {}
    for i in range(1, 901):
        stims[f"cell {i}"] = aux.random_stim_maker(total_time).tolist()
    for i in range(901, 951):
        stims[f"cell {i}"] = aux.repetitive_stim_maker(
            num_repeat=4,
            total_time=total_time,
            off_first=True,
        ).tolist()
    for i in range(951, 1001):
        stims[f"cell {i}"] = aux.repetitive_stim_maker(
            num_repeat=4,
            total_time=total_time,
            off_first=False,
        ).tolist()
    return stims


def default_training_data_root():
    project_parent = Path(__file__).resolve().parents[3]
    return project_parent / "assets" / "training_data"


def default_models_root():
    project_parent = Path(__file__).resolve().parents[3]
    return project_parent / "assets" / "models"


def _resolve_model_config(model):
    if model is None or model is False:
        return None
    if isinstance(model, str):
        return {"type": model}
    if isinstance(model, dict):
        model_config = dict(model)
        model_config.setdefault(
            "type",
            model_config.get("model", model_config.get("name", model_config.get("model_type"))),
        )
        if model_config["type"] is None:
            raise ValueError("Model config must include a model type/name.")
        return model_config
    raise ValueError("model must be None, a model name string, or a model config dictionary.")


def _train_and_save_model(model_config, cells, data_config, run_stamp):
    model_type = str(model_config["type"]).lower()
    if model_type in {"regressor", "regression", "regression_forecaster"}:
        return _train_and_save_regressor(model_config, cells, data_config, run_stamp)
    if model_type in {"transformer", "transformer_forecaster"}:
        raise NotImplementedError("Transformer forecaster training is not implemented yet.")
    raise NotImplementedError(f"Model type {model_config['type']!r} is not implemented yet.")


def _train_and_save_regressor(model_config, cells, data_config, run_stamp):
    from aisam.comptools.regression_forecaster import train_regression_forecaster

    hyperparams = dict(model_config.get("hyperparameters", model_config.get("hyperparams", {})))
    for key, value in model_config.items():
        if key not in {"type", "model", "name", "model_type", "hyperparameters", "hyperparams", "output_root"}:
            hyperparams.setdefault(key, value)

    hyperparams.setdefault("past_feature_window", 20)
    hyperparams.setdefault("future_window", 1)
    hyperparams.setdefault("output_species", "F")

    forecaster, metrics, dataset = train_regression_forecaster(cells, **hyperparams)

    model_root = Path(model_config.get("output_root", default_models_root()))
    model_dir = model_root / f"regressor_{run_stamp}"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.pkl"
    with open(model_path, "wb") as f:
        dill.dump(forecaster, f)

    model_config_dump = {
        "created_at": run_stamp,
        "model_type": "regressor",
        "model_file": str(model_path),
        "model_hyperparameters": _json_safe(hyperparams),
        "metrics": _json_safe(metrics),
        "data": {
            "simulation_file": data_config["simulation_file"],
            "simulation_config": data_config,
            "train_sequences": len(dataset["train_sequences"]),
            "validation_sequences": len(dataset["validation_sequences"]),
            "train_windows": int(dataset["X_train"].shape[0]),
            "validation_windows": int(dataset["X_validation"].shape[0])
            if "X_validation" in dataset
            else 0,
        },
    }

    model_config_path = model_dir / "config.json"
    with open(model_config_path, "w") as f:
        json.dump(model_config_dump, f, indent=2)

    return {
        "model_dir": model_dir,
        "model_path": model_path,
        "config_path": model_config_path,
        "metrics": metrics,
        "config": model_config_dump,
    }


def _load_config(config):
    if isinstance(config, (str, PathLike)):
        with open(config, "r") as f:
            return json.load(f)
    if isinstance(config, dict):
        return dict(config)
    raise ValueError("config must be a dictionary or a JSON file path.")


def _load_params_from_config(config):
    params = config.get("params", config.get("circuit_parameters"))
    if params is not None:
        if not isinstance(params, dict):
            raise ValueError("params/circuit_parameters must be a dictionary.")
        return dict(params)

    params_path = config.get("params_path")
    if params_path is None:
        root_folder = config.get("root_folder")
        if root_folder is not None:
            params_path = Path(root_folder) / "simulation_params.json"

    if params_path is None:
        raise ValueError("Provide params, circuit_parameters, params_path, or root_folder.")

    return aux.load_params(params_path)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, PathLike):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value
