import json
from datetime import datetime
from os import PathLike
from pathlib import Path

import dill
import numpy as np

from aisam.model import sims
from aisam.utils import aux


def training(config, model=None, total_cell=None, noisy_total_cells=None):
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
    - total_cell, total_cells, or num_cells: standard simulated cells, minimum 200
    - noisy_total_cells: cells per noisy simulation, defaults to 350
    - sample_interval_minutes: optional forecaster preprocessing sample interval
    - model: optional model config. Supported model type: "regressor"

    `model` can also be passed directly as a string or dictionary and overrides
    config["model"].

    Returns a dictionary of saved paths and run metadata.
    """
    result = run_training_simulation(
        config,
        total_cell=total_cell,
        noisy_total_cells=noisy_total_cells,
        include_cells=True,
    )
    config = result["user_config"]
    run_stamp = result["run_stamp"]

    model_config = _resolve_model_config(model if model is not None else config.get("model"))
    if model_config is not None:
        model_result = _train_and_save_model(
            model_config=model_config,
            cells=result["cells"],
            data_config=result["config"],
            run_stamp=run_stamp,
        )
        result["model"] = model_result

    result.pop("cells", None)
    return result


def run_training_simulation(
    root_folder,
    label=None,
    total_cell=None,
    noisy_sims=None,
    noisy_total_cells=None,
    temperatures=None,
    output_root=None,
    random_seed=None,
    include_cells=False,
):
    """
    Generate standard training data through simulation.

    `root_folder` can be a directory containing `config.json` and
    `simulation_params.json`, a config JSON path, or a config dictionary. The
    standard run uses a configurable stimulation panel. Optional noisy runs use
    sampled parameter dictionaries and a configurable 350-cell panel by default.

    Config keys:
    - circuit or label: circuit name, defaults to "CcaSR"
    - root_folder: directory with config/parameter files when config is a dict
    - t_max: stimulation length
    - sampling: simulation sampling, defaults to params["sampling"] or 10
    - total_cell, total_cells, or num_cells: standard simulated cells, minimum 200
    - num_realizations: number of trajectories per cell, defaults to 1
    - noisy_sims: optional number of noisy simulations
    - noisy_total_cells: cells per noisy simulation, defaults to 350
    - temperatures: optional list used to divide noisy simulations by noise level
    - output_root: optional explicit override for the run output directory
    - include_cells: include the in-memory standard Cells object in the return
    - progress: show a per-cell progress bar during simulation

    Returns a dictionary with saved paths, metadata, and noisy run results.
    """
    config, source_root, source_kind = _load_simulation_config_from_root(root_folder)

    if label is not None:
        config["label"] = label
    if output_root is not None:
        config["output_root"] = str(output_root)
    if random_seed is not None:
        config["random_seed"] = random_seed
    if total_cell is not None:
        config["total_cell"] = total_cell
    if noisy_sims is not None:
        config["noisy_sims"] = noisy_sims
    if noisy_total_cells is not None:
        config["noisy_total_cells"] = noisy_total_cells
    if temperatures is not None:
        config["temperatures"] = temperatures

    label = config.get("label", config.get("circuit", "CcaSR"))
    total_cells = _resolve_total_cells(config, default=1000)
    noisy_total_cells = _resolve_total_cells(
        config,
        keys=("noisy_total_cells", "noisy_total_cell", "num_noisy_cells"),
        default=350,
    )

    params = _load_params_from_config(config, source_root=source_root)
    if config.get("t_max", params.get("t_max")) is None:
        raise ValueError("Provide t_max in the config or parameter dictionary.")

    t_max = int(config.get("t_max", params.get("t_max")))
    sampling = int(config.get("sampling", params.get("sampling", 10)))
    num_realizations = int(config.get("num_realizations", 1))
    progress = bool(config.get("progress", True))
    params.setdefault("t_max", t_max)
    params.setdefault("sampling", sampling)

    random_seed = config.get("random_seed")
    if random_seed is not None:
        np.random.seed(int(random_seed))

    stims = build_standard_stims(t_max, total_cells=total_cells)
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_root = _resolve_training_data_root(
        config=config,
        source_root=source_root,
        source_kind=source_kind,
        explicit_output_root=output_root,
    )
    run_dir = run_root / f"{label}_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    standard_result = _run_and_save_simulation(
        params=params,
        label=label,
        stims=stims,
        num_cells=total_cells,
        num_realizations=num_realizations,
        t_max=t_max,
        sampling=sampling,
        run_dir=run_dir,
        run_stamp=run_stamp,
        user_config=config,
        progress=progress,
        desc=f"{label} standard simulation",
    )

    result = {
        **standard_result,
        "run_stamp": run_stamp,
        "user_config": config,
        "noisy": [],
    }
    if include_cells:
        result["cells"] = standard_result["cells"]
    else:
        result.pop("cells", None)

    noisy_count = int(config.get("noisy_sims", 0))
    if noisy_count > 0:
        noisy_stims = build_standard_stims(t_max, total_cells=noisy_total_cells)
        temp_schedule = assign_temperatures(
            noisy_count,
            config.get("temperatures", [0.1]),
        )
        noisy_root = run_dir / "noisy"
        noisy_root.mkdir(parents=True, exist_ok=True)

        for i, temp in enumerate(temp_schedule, start=1):
            sampled_params = aux.sample_noisy_params_from_dict(params, temperature=temp)
            sim_dir = noisy_root / f"sim_{i}"
            noisy_result = _run_and_save_simulation(
                params=sampled_params,
                label=label,
                stims=noisy_stims,
                num_cells=noisy_total_cells,
                num_realizations=num_realizations,
                t_max=t_max,
                sampling=sampling,
                run_dir=sim_dir,
                run_stamp=run_stamp,
                user_config=config,
                temperature=temp,
                parent_run_dir=run_dir,
                progress=progress,
                desc=f"{label} noisy simulation {i}/{noisy_count}",
            )
            noisy_result.pop("cells", None)
            result["noisy"].append(noisy_result)

    return result


def build_standard_stims(total_time, total_cells=1000):
    total_cells = _validate_total_cells(total_cells)
    random_cells = total_cells - 100
    red_first_start = random_cells + 1
    green_first_start = random_cells + 51

    stims = {}
    for i in range(1, random_cells + 1):
        stims[f"cell {i}"] = aux.random_stim_maker(total_time).tolist()
    for i in range(red_first_start, green_first_start):
        stims[f"cell {i}"] = aux.repetitive_stim_maker(
            num_repeat=4,
            total_time=total_time,
            off_first=True,
        ).tolist()
    for i in range(green_first_start, total_cells + 1):
        stims[f"cell {i}"] = aux.repetitive_stim_maker(
            num_repeat=4,
            total_time=total_time,
            off_first=False,
        ).tolist()
    return stims


def assign_temperatures(num_sims, temperatures):
    if num_sims <= 0:
        return []
    if not temperatures:
        return [0.1] * num_sims

    base = num_sims // len(temperatures)
    rem = num_sims % len(temperatures)
    assigned = []
    for i, temp in enumerate(temperatures):
        count = base + (1 if i < rem else 0)
        assigned.extend([float(temp)] * count)
    return assigned


def stimulation_cell_ranges(total_cells):
    total_cells = _validate_total_cells(total_cells)
    random_cells = total_cells - 100
    return {
        "total_cells": total_cells,
        "random_stimulation_cells": [1, random_cells],
        "repetitive_stimulation_cells_red_first": [random_cells + 1, random_cells + 50],
        "repetitive_stimulation_cells_green_first": [random_cells + 51, total_cells],
    }


def _resolve_total_cells(config, keys=("total_cell", "total_cells", "num_cells"), default=1000):
    for key in keys:
        if key in config and config[key] is not None:
            return _validate_total_cells(config[key])
    return _validate_total_cells(default)


def _validate_total_cells(total_cells):
    total_cells = int(total_cells)
    if total_cells < 200:
        raise ValueError("total_cell must be at least 200.")
    return total_cells


def default_training_data_root():
    return dated_assets_root() / "training_data"


def dated_assets_root(date_stamp=None):
    project_parent = Path(__file__).resolve().parents[3]
    date_stamp = date_stamp or datetime.now().strftime("%y%m%d")
    return project_parent / "assets" / date_stamp


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
    hyperparams.setdefault("sampling", data_config["simulated_cells"]["sampling"])
    sample_interval = _sample_interval_from_config(data_config.get("user_config", {}))
    if sample_interval is not None:
        hyperparams.setdefault("sample_interval_minutes", sample_interval)

    forecaster, metrics, dataset = train_regression_forecaster(cells, **hyperparams)

    model_root = Path(
        model_config.get(
            "output_root",
            Path(data_config["simulation_file"]).parent / "models",
        )
    )
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


def _sample_interval_from_config(config):
    for key in (
        "sample_interval_minutes",
        "training_sample_interval_minutes",
        "forecaster_sample_interval_minutes",
        "sample_rate_minutes",
        "interval_rate",
    ):
        if key in config and config[key] is not None:
            return config[key]
    return None


def _run_and_save_simulation(
    params,
    label,
    stims,
    num_cells,
    num_realizations,
    t_max,
    sampling,
    run_dir,
    run_stamp,
    user_config,
    temperature=None,
    parent_run_dir=None,
    progress=True,
    desc=None,
):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    xpt = sims.experiment(params, label)
    xpt.init_exp(num_cells)
    xpt.run_training_sim(stims, num_realizations, progress=progress, desc=desc)

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
        "run_dir": str(run_dir),
        "circuit_parameters": _json_safe(params),
        "simulation_file": str(simulation_path),
        "simulation_params_file": str(params_path),
        "stims_file": str(stims_path),
        "simulated_cells": {
            **stimulation_cell_ranges(num_cells),
            "num_realizations": num_realizations,
            "t_max": t_max,
            "sampling": sampling,
        },
        "user_config": _json_safe(user_config),
    }
    if temperature is not None:
        config_dump["temperature"] = float(temperature)
    if parent_run_dir is not None:
        config_dump["parent_run_dir"] = str(parent_run_dir)

    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dump, f, indent=2)

    return {
        "run_dir": run_dir,
        "simulation_path": simulation_path,
        "params_path": params_path,
        "stims_path": stims_path,
        "config_path": config_path,
        "config": config_dump,
        "cells": xpt.Cells,
    }


def _load_config(config):
    if isinstance(config, (str, PathLike)):
        with open(config, "r") as f:
            return json.load(f)
    if isinstance(config, dict):
        return dict(config)
    raise ValueError("config must be a dictionary or a JSON file path.")


def _resolve_training_data_root(config, source_root, source_kind, explicit_output_root=None):
    if explicit_output_root is not None:
        return Path(explicit_output_root)
    if source_kind == "config_path":
        return dated_assets_root() / "training_data"
    if source_root is None:
        source_root = dated_assets_root()
    return Path(source_root) / "training_data"


def _load_simulation_config_from_root(root_folder):
    if isinstance(root_folder, dict):
        config = dict(root_folder)
        source_root = Path(config["root_folder"]) if "root_folder" in config else None
        return config, source_root, "dict"

    path = Path(root_folder)
    if path.is_dir():
        config_path = path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        config.setdefault("root_folder", str(path))
        return config, path, "root_dir"

    with open(path, "r") as f:
        config = json.load(f)
    source_root = path.parent
    config.setdefault("root_folder", str(source_root))
    return config, source_root, "config_path"


def _load_params_from_config(config, source_root=None):
    params = config.get("params", config.get("circuit_parameters"))
    if params is not None:
        if not isinstance(params, dict):
            raise ValueError("params/circuit_parameters must be a dictionary.")
        return dict(params)

    params_path = config.get("params_path")
    if params_path is None:
        root_folder = config.get("root_folder") or source_root
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
