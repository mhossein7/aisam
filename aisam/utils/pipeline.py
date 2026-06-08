import json
from pathlib import Path

from aisam.model import defaults
from aisam.utils import aux
from aisam.utils.forecaster_training import (
    _resolve_model_config,
    train_forecaster,
)
from aisam.utils.simulation import (
    run_sanity_check_simulation,
    run_training_simulation,
)


def training(
    config,
    model=None,
    total_cell=None,
    noisy_total_cells=None,
    include_noisy="none",
    include_noisy_periodic="none",
    include_main_periodic=None,
):
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
        include_cells=False,
    )
    config = result["user_config"]

    model_config = _resolve_model_config(model if model is not None else config.get("model"))
    if model_config is not None:
        include_main_periodic = (
            include_main_periodic
            if include_main_periodic is not None
            else config.get("include_main_periodic", "train")
        )
        model_result = train_forecaster(
            path=result["simulation_path"],
            config=model_config,
            visualization=bool(model_config.get("visualization", False)),
            random_state=config.get("random_seed"),
            include_noisy=include_noisy,
            include_noisy_periodic=include_noisy_periodic,
            include_main_periodic=include_main_periodic,
        )
        result["model"] = model_result

    return result


def run_recipe(root_folder=None, progress=None):
    """
    Run a complete experiment described by `recipe.json`.

    The recipe is read from `root_folder` or the current working directory. If
    `config.json` and/or `simulation_params.json` are present in the same root,
    they override recipe/default simulation settings.
    """
    recipe_config = load_experiment_recipe(root_folder)
    config = recipe_config["config"]
    recipe = recipe_config["recipe"]
    root = recipe_config["root_folder"]
    if progress is not None:
        config["progress"] = bool(progress)
    mode = _recipe_mode(recipe)

    result = {
        "root_folder": root,
        "recipe_path": recipe_config["recipe_path"],
        "mode": mode or "custom",
        "config": config,
        "sanity": None,
        "simulation": None,
        "model": None,
    }

    include_sanity = _recipe_bool(recipe, "include_sanity_check", "sanity_check", default=False)
    sanity_only = _recipe_bool(recipe, "sanity_only", "only_sanity_check", default=False)
    include_training_simulation = _recipe_bool(
        recipe,
        "include_training_simulation",
        "include_simulation",
        "run_training_simulation",
        default=not sanity_only,
    )
    include_model_training = _recipe_bool(
        recipe,
        "include_model_training",
        "train_model",
        "include_training",
        default=False,
    )
    if mode == "sanity":
        include_sanity = True
        sanity_only = True
        include_training_simulation = False
        include_model_training = False
    elif mode == "simulation":
        include_training_simulation = True
        include_model_training = False
    elif mode == "training":
        include_training_simulation = False
        include_model_training = True
    elif mode == "full":
        include_training_simulation = True
        include_model_training = True

    if include_sanity:
        result["sanity"] = run_sanity_check_simulation(config, progress=config.get("progress", True))

    if sanity_only and not include_model_training:
        return result

    training_path = None
    if include_training_simulation:
        simulation_result = run_training_simulation(config, include_cells=False)
        result["simulation"] = simulation_result
        training_path = simulation_result["simulation_path"]

    if include_model_training:
        model_config = _resolve_model_config(config.get("model", {"type": "regressor"}))
        if model_config is None:
            raise ValueError("Recipe requested model training but did not define a forecaster model.")
        include_repetitive_training = bool(config.get("include_repetitive_stims_in_training", True))
        include_repetitive_eval = bool(config.get("include_repetitive_eval", True))
        include_noisy = config.get("include_noisy", recipe.get("include_noisy", "none"))
        include_noisy_periodic = config.get(
            "include_noisy_periodic",
            recipe.get("include_noisy_periodic", "none"),
        )
        training_path = training_path or _recipe_training_data_path(config, recipe, root)
        model_result = train_forecaster(
            path=training_path,
            config=model_config,
            visualization=bool(model_config.get("visualization", False)),
            random_state=config.get("random_seed"),
            include_noisy=include_noisy,
            include_noisy_periodic=include_noisy_periodic,
            include_main_periodic="train"
            if include_repetitive_training
            else ("eval" if include_repetitive_eval else "none"),
        )
        result["model"] = model_result

    return result


def load_experiment_recipe(root_folder=None):
    """
    Load `recipe.json` and resolve it into the standard simulation config.

    Priority for simulation parameters is:
    model defaults < recipe parameters < config.json parameters <
    root/simulation_params.json.
    """
    root = _resolve_recipe_root(root_folder)
    recipe_path = root / "recipe.json"
    if not recipe_path.exists():
        raise FileNotFoundError(f"No recipe.json found in {root}.")
    with open(recipe_path, "r") as f:
        recipe = json.load(f)
    if not isinstance(recipe, dict):
        raise ValueError("recipe.json must contain a JSON object.")

    config_path = root / "config.json"
    file_config = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            file_config = json.load(f)

    recipe_base = _config_from_recipe(recipe)
    config = {**recipe_base, **file_config}
    config["root_folder"] = str(root)

    legacy_simulation_model = config.pop("simulation_model", recipe.get("simulation_model"))
    solver_choice = _first_present(
        config,
        ("solver", "sovler", "simulation_solver", "simulator"),
    )
    if solver_choice is None and legacy_simulation_model is not None:
        if defaults.is_supported_circuit(legacy_simulation_model):
            solver_choice = None
        else:
            solver_choice = legacy_simulation_model
    raw_circuit = config.get("circuit", config.get("label", recipe.get("circuit")))
    circuit = raw_circuit
    if defaults.is_supported_circuit(legacy_simulation_model):
        circuit = legacy_simulation_model
    if circuit is None:
        circuit = "ccasr"
    circuit = defaults.normalize_circuit_name(circuit)
    config["circuit"] = circuit
    if config.get("label") is None or defaults.is_supported_circuit(config.get("label")):
        config["label"] = circuit

    params = defaults.default_circuit_params(raw_circuit or circuit)
    if defaults.is_measurement_noise_variant(raw_circuit):
        params["measurement_noise"] = True
    params.update(_recipe_params(recipe))
    file_params = file_config.get("params", file_config.get("circuit_parameters"))
    if isinstance(file_params, dict):
        params.update(file_params)

    root_params_path = root / "simulation_params.json"
    if root_params_path.exists():
        params.update(aux.load_params(root_params_path))
        config["params_path"] = str(root_params_path)
    elif file_config.get("params_path"):
        params_path = Path(file_config["params_path"])
        if not params_path.is_absolute():
            params_path = root / params_path
        params.update(aux.load_params(params_path))
        config["params_path"] = str(params_path)

    params = defaults.clean_circuit_params(circuit, params)
    config["params"] = params
    config["circuit_parameters"] = params
    config.setdefault("t_max", params.get("t_max", 960))
    config.setdefault("sampling", params.get("sampling", 10))
    params["t_max"] = config["t_max"]
    params["sampling"] = config["sampling"]
    config.setdefault("interval_rate", recipe.get("interval_rate", recipe.get("sample_interval_minutes", 5)))
    config.setdefault("total_cell", recipe.get("total_cell", recipe.get("total_cells", recipe.get("num_cells", 1000))))
    config.setdefault("noisy_total_cells", recipe.get("noisy_total_cells", 350))
    config.setdefault("num_realizations", recipe.get("num_realizations", 1))
    config.setdefault("noisy_sims", recipe.get("noisy_sims", 0))
    config.setdefault("progress", recipe.get("progress", True))
    config.setdefault("include_noisy", recipe.get("include_noisy", "none"))
    config.setdefault("include_noisy_periodic", recipe.get("include_noisy_periodic", "none"))

    include_periodic_training = _recipe_bool(
        recipe,
        "include_periodic_stims_in_training",
        "include_repetitive_stims_in_training",
        default=True,
    )
    include_periodic_validation = _recipe_bool(
        recipe,
        "include_periodic_stims_in_validation",
        "include_repetitive_stims_in_validation",
        "include_repetitive_eval",
        default=True,
    )
    config.setdefault("include_repetitive_stims_in_training", include_periodic_training)
    config.setdefault("include_repetitive_eval", include_periodic_validation)
    config.setdefault(
        "include_repetitive_stims",
        bool(config["include_repetitive_stims_in_training"] or config["include_repetitive_eval"]),
    )

    config["solver"] = _resolve_solver_for_circuit(solver_choice, circuit)
    config.pop("sovler", None)

    forecaster_model = config.get(
        "forecaster_model",
        recipe.get(
            "forecaster_model",
            recipe.get("model", recipe.get("model_type", "regressor")),
        ),
    )
    config["model"] = _recipe_model_config(forecaster_model, recipe)

    return {
        "root_folder": root,
        "recipe_path": recipe_path,
        "config_path": config_path if config_path.exists() else None,
        "params_path": root_params_path if root_params_path.exists() else config.get("params_path"),
        "recipe": recipe,
        "config": config,
    }


def _resolve_recipe_root(root_folder):
    if root_folder is None:
        return Path.cwd()
    path = Path(root_folder)
    if path.is_file():
        if path.name != "recipe.json":
            raise ValueError("Recipe path inputs must point to recipe.json.")
        return path.parent
    return path


def _config_from_recipe(recipe):
    config = {}
    key_map = {
        "circuit": ("circuit", "circuit_type", "type_of_circuit"),
        "label": ("label", "experiment_label"),
        "t_max": ("t_max", "duration_minutes"),
        "sampling": ("sampling", "simulation_sampling"),
        "interval_rate": ("interval_rate", "sample_interval_minutes", "sample_rate_minutes"),
        "total_cell": ("total_cell", "total_cells", "num_cells"),
        "noisy_total_cells": ("noisy_total_cells", "num_noisy_cells"),
        "num_realizations": ("num_realizations", "num_realization"),
        "noisy_sims": ("noisy_sims", "num_noisy_sims"),
        "temperatures": ("temperatures", "noise_temperatures"),
        "random_seed": ("random_seed", "seed"),
        "progress": ("progress",),
        "mode": ("mode", "experiment_mode", "run_mode", "pipeline_mode"),
        "solver": ("solver", "sovler", "simulation_solver", "simulator"),
        "simulation_path": ("simulation_path", "simulation_file", "simulation_data_file", "data_path"),
        "output_root": ("output_root",),
        "save_parquet": ("save_parquet",),
        "save_pickle": ("save_pickle",),
    }
    for target, aliases in key_map.items():
        value = _first_present(recipe, aliases)
        if value is not None:
            config[target] = value

    simulation = recipe.get("simulation")
    if isinstance(simulation, dict):
        for key, value in simulation.items():
            if key not in {"parameters", "params", "circuit_parameters"}:
                config.setdefault(key, value)

    return config


def _recipe_params(recipe):
    params = {}
    for key in ("params", "parameters", "circuit_parameters", "simulation_params"):
        value = recipe.get(key)
        if isinstance(value, dict):
            params.update(value)
    simulation = recipe.get("simulation")
    if isinstance(simulation, dict):
        for key in ("params", "parameters", "circuit_parameters", "simulation_params"):
            value = simulation.get(key)
            if isinstance(value, dict):
                params.update(value)
    return params


def _recipe_model_config(forecaster_model, recipe):
    if forecaster_model is None or forecaster_model is False:
        return None
    if isinstance(forecaster_model, dict):
        model_config = dict(forecaster_model)
    else:
        model_config = {"type": forecaster_model}

    for key in ("model_hyperparameters", "forecaster_hyperparameters", "hyperparameters", "hyperparams"):
        value = recipe.get(key)
        if isinstance(value, dict):
            model_config.setdefault("hyperparameters", {}).update(value)

    visualization = _first_present(recipe, ("visualization", "include_visualization", "plot_model_performance"))
    if visualization is not None:
        model_config["visualization"] = bool(visualization)
    return model_config


def _recipe_bool(recipe, *keys, default=False):
    value = _first_present(recipe, keys)
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _recipe_mode(recipe):
    value = _first_present(recipe, ("mode", "experiment_mode", "run_mode", "pipeline_mode"))
    if value is None:
        return None
    value = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "sim": "simulation",
        "simulate": "simulation",
        "train": "training",
        "model_training": "training",
        "pipeline": "full",
        "full_pipeline": "full",
        "simulation_and_training": "full",
        "sanity_check": "sanity",
    }
    value = aliases.get(value, value)
    if value not in {"sanity", "simulation", "training", "full"}:
        raise ValueError("Recipe mode must be one of: sanity, simulation, training, full.")
    return value


def _recipe_training_data_path(config, recipe, root):
    for mapping in (config, recipe):
        value = _first_present(
            mapping,
            ("simulation_path", "simulation_file", "simulation_data_file", "data_path"),
        )
        if value:
            path = Path(value)
            return path if path.is_absolute() else Path(root) / path
    return root


def _first_present(mapping, keys):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _resolve_solver_for_circuit(solver, circuit):
    solver = solver or defaults.default_solver_for_circuit(circuit)
    if defaults.is_supported_circuit(solver):
        raise ValueError(
            f"{solver!r} is a circuit/model name. Put it in the recipe `circuit` field "
            "and use `solver` for the simulation backend."
        )
    return defaults.validate_solver_for_circuit(solver, circuit)
