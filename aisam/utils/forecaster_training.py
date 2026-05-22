import json
from datetime import datetime
from os import PathLike
from pathlib import Path

import dill
import numpy as np

from aisam.utils.simulation import (
    _json_safe,
    _sample_interval_from_config,
    run_training_simulation,
    stimulation_cell_ranges,
)


def train_forecaster_from_simulation(
    simulation_paths,
    model=None,
    output_root=None,
    visualization=False,
    random_state=None,
    include_noisy="none",
    include_noisy_periodic="none",
    include_main_periodic="none",
):
    """
    Train a forecaster model from one or more saved `simulation.pkl` files.

    When multiple simulation files are supplied, cells are merged with
    file-specific cell ids so windows from same-numbered cells in different runs
    are not grouped together.
    """
    if (
        _policy_enabled(include_noisy)
        or _policy_enabled(include_noisy_periodic)
        or _policy_enabled(include_main_periodic)
    ):
        return train_forecaster_from_simulation_config(
            path=simulation_paths,
            config=model,
            output_root=output_root,
            visualization=visualization,
            random_state=random_state,
            include_noisy=include_noisy,
            include_noisy_periodic=include_noisy_periodic,
            include_main_periodic=include_main_periodic,
        )

    from aisam.comptools.regression_forecaster import train_regression_forecaster

    paths = _coerce_simulation_paths(simulation_paths)
    cells = _load_and_merge_cells(paths)
    run_configs = [_load_run_config_from_simulation_path(path) for path in paths]
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model_config = _resolve_model_config(model or {"type": "regressor"})
    model_type = str(model_config["type"]).lower()
    if model_type not in {"regressor", "regression", "regression_forecaster"}:
        raise NotImplementedError(f"Model type {model_config['type']!r} is not implemented yet.")

    hyperparams = _forecaster_hyperparams_from_model_config(model_config)
    first_config = next((cfg for cfg in run_configs if cfg), {})
    hyperparams.setdefault("past_feature_window", 20)
    hyperparams.setdefault("future_window", 1)
    hyperparams.setdefault("output_species", "F")
    if random_state is not None:
        hyperparams.setdefault("random_state", random_state)

    forecaster, metrics, dataset = train_regression_forecaster(cells, **hyperparams)
    _assign_saved_sampling_metadata(forecaster, first_config)

    model_root = Path(output_root or model_config.get("output_root") or (paths[0].parent / "models"))
    model_dir = model_root / f"regressor_{run_stamp}"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.pkl"
    with open(model_path, "wb") as f:
        dill.dump(forecaster, f)

    artifacts = {}
    if "X_validation" in dataset:
        artifacts = _save_evaluation_artifacts(
            forecaster=forecaster,
            dataset=dataset,
            output_dir=model_dir / "figures",
            output_species=hyperparams.get("output_species", "F"),
            random_state=random_state,
            title="validation",
        )
    performance_path = _save_performance_info(
        model_dir / "performance.json",
        metrics=metrics,
        dataset=dataset if "X_validation" in dataset else None,
        label="validation",
    )

    model_config_dump = {
        "created_at": run_stamp,
        "model_type": "regressor",
        "model_file": str(model_path),
        "model_hyperparameters": _json_safe(hyperparams),
        "metrics": _json_safe(metrics),
        "data": {
            "simulation_files": [str(path) for path in paths],
            "simulation_configs": _json_safe(run_configs),
            "train_sequences": len(dataset["train_sequences"]),
            "validation_sequences": len(dataset["validation_sequences"]),
            "train_windows": int(dataset["X_train"].shape[0]),
            "validation_windows": int(dataset["X_validation"].shape[0])
            if "X_validation" in dataset
            else 0,
        },
        "performance_file": str(performance_path),
        "visualization_files": {key: str(value) for key, value in artifacts.get("figures", {}).items()},
    }

    model_config_path = model_dir / "config.json"
    with open(model_config_path, "w") as f:
        json.dump(model_config_dump, f, indent=2)

    return {
        "model_dir": model_dir,
        "model_path": model_path,
        "config_path": model_config_path,
        "metrics": metrics,
        "figures": artifacts.get("figures", {}),
        "performance_path": performance_path,
        "config": model_config_dump,
    }


def train_forecaster_from_simulation_config(
    path=None,
    config=None,
    output_root=None,
    visualization=False,
    random_state=None,
    include_noisy="none",
    include_noisy_periodic="eval",
    include_main_periodic="eval",
    label=None,
):
    """
    Train a forecaster from existing simulation data with explicit policies.

    Policies are "train", "eval", or "none":
    - include_main_periodic controls main-run repetitive stim cells.
    - include_noisy controls noisy-run random stim cells.
    - include_noisy_periodic controls noisy-run repetitive stim cells.
    """
    from aisam.comptools.regression_forecaster import (
        RegressionForecaster,
        make_window_dataset,
        regression_metrics,
        split_sequences_by_cell,
    )

    main_paths = _resolve_main_simulation_paths(path)
    noisy_paths = _discover_noisy_simulation_paths(main_paths[0])
    model_config = _resolve_model_config(config or {"type": "regressor"})
    if model_config is None:
        raise ValueError("A model config is required for forecaster training.")
    model_type = str(model_config["type"]).lower()
    if model_type not in {"regressor", "regression", "regression_forecaster"}:
        raise NotImplementedError(f"Model type {model_config['type']!r} is not implemented yet.")

    if visualization is None:
        visualization = bool(model_config.get("visualization", False))
    else:
        visualization = bool(visualization)

    include_noisy = _normalize_policy(include_noisy)
    include_noisy_periodic = _normalize_policy(include_noisy_periodic)
    include_main_periodic = _normalize_policy(include_main_periodic)

    hyperparams = _forecaster_hyperparams_from_model_config(model_config)
    hyperparams.setdefault("past_feature_window", 20)
    hyperparams.setdefault("future_window", 1)
    hyperparams.setdefault("output_species", "F")
    if random_state is not None:
        hyperparams.setdefault("random_state", random_state)
    validation_fraction = float(hyperparams.get("validation_fraction", 0.2))
    if validation_fraction <= 0 or validation_fraction >= 1:
        raise ValueError("validation_fraction must be in the range (0, 1).")

    main_random, main_repetitive, main_configs = _sequences_by_stim_group_from_paths(
        main_paths,
        model_config=model_config,
        run_label="main",
    )
    train_sequences, eval_sequences = split_sequences_by_cell(
        main_random,
        validation_fraction=validation_fraction,
        random_state=hyperparams.get("random_state"),
    )
    if include_main_periodic == "train":
        train_sequences.extend(main_repetitive)
    elif include_main_periodic == "eval":
        eval_sequences.extend(main_repetitive)

    noisy_eval_groups = []
    noisy_train_sequences = []
    if noisy_paths and (_policy_enabled(include_noisy) or _policy_enabled(include_noisy_periodic)):
        for noisy_path in noisy_paths:
            noisy_random, noisy_repetitive, noisy_configs = _sequences_by_stim_group_from_paths(
                [noisy_path],
                model_config=model_config,
                run_label=noisy_path.parent.name,
            )
            group_eval = []
            if include_noisy == "train":
                noisy_train_sequences.extend(noisy_random)
            elif include_noisy == "eval":
                group_eval.extend(noisy_random)
            if include_noisy_periodic == "train":
                noisy_train_sequences.extend(noisy_repetitive)
            elif include_noisy_periodic == "eval":
                group_eval.extend(noisy_repetitive)
            if group_eval:
                eval_sequences.extend(group_eval)
                noisy_eval_groups.append(
                    {
                        "path": noisy_path,
                        "configs": noisy_configs,
                        "sequences": group_eval,
                    }
                )
    train_sequences.extend(noisy_train_sequences)

    if not train_sequences:
        raise ValueError("No training sequences were selected. Check train/eval/none policies.")
    if not eval_sequences:
        raise ValueError("No evaluation sequences were selected. At least a random holdout is required.")

    window_args = _window_args_from_hyperparams(hyperparams)
    X_train, y_train, train_meta = make_window_dataset(train_sequences, **window_args)
    X_eval, y_eval, eval_meta = make_window_dataset(eval_sequences, **window_args)

    forecaster = RegressionForecaster(
        past_feature_window=window_args["past_feature_window"],
        future_window=window_args["future_window"],
        past_input_window=window_args["past_input_window"],
        future_input_window=window_args["future_input_window"],
        regressor=hyperparams.get("regressor"),
        normalize=hyperparams.get("normalize", True),
    )
    forecaster.fit(X_train, y_train)
    eval_predictions = forecaster.predict(X_eval)
    metrics = {
        "train": forecaster.evaluate(X_train, y_train),
        "evaluation": regression_metrics(y_eval, eval_predictions),
    }
    first_config = next((cfg for cfg in main_configs if cfg), {})
    _assign_saved_sampling_metadata(forecaster, first_config)

    dataset = {
        "X_train": X_train,
        "y_train": y_train,
        "train_meta": train_meta,
        "train_sequences": train_sequences,
        "validation_sequences": eval_sequences,
        "X_validation": X_eval,
        "y_validation": y_eval,
        "validation_meta": eval_meta,
        "validation_predictions": eval_predictions,
    }

    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_root = Path(output_root or model_config.get("output_root") or (main_paths[0].parent / "models"))
    model_label = label or _model_label(model_config, include_noisy, include_main_periodic)
    model_dir = model_root / f"{model_label}_{run_stamp}"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.pkl"
    with open(model_path, "wb") as f:
        dill.dump(forecaster, f)

    artifacts = _save_evaluation_artifacts(
        forecaster=forecaster,
        dataset=dataset,
        output_dir=model_dir / "figures",
        output_species=hyperparams.get("output_species", "F"),
        random_state=hyperparams.get("random_state"),
        title="evaluation",
    )
    performance_path = _save_performance_info(
        model_dir / "performance.json",
        metrics=metrics,
        dataset=dataset,
        label="evaluation",
    )

    noisy_results = []
    for noisy_group in noisy_eval_groups:
        noisy_dataset, noisy_metrics = _evaluate_sequences(
            forecaster=forecaster,
            sequences=noisy_group["sequences"],
            hyperparams=hyperparams,
        )
        noisy_dir = model_dir / "noisy_evaluation" / noisy_group["path"].parent.name
        noisy_artifacts = _save_evaluation_artifacts(
            forecaster=forecaster,
            dataset=noisy_dataset,
            output_dir=noisy_dir / "figures",
            output_species=hyperparams.get("output_species", "F"),
            random_state=hyperparams.get("random_state"),
            title=noisy_group["path"].parent.name,
        )
        noisy_performance_path = _save_performance_info(
            noisy_dir / "performance.json",
            metrics={"evaluation": noisy_metrics},
            dataset=noisy_dataset,
            label=noisy_group["path"].parent.name,
        )
        noisy_results.append(
            {
                "simulation_file": noisy_group["path"],
                "performance_path": noisy_performance_path,
                "metrics": noisy_metrics,
                "figures": noisy_artifacts.get("figures", {}),
                "num_sequences": len(noisy_group["sequences"]),
            }
        )

    model_config_dump = {
        "created_at": run_stamp,
        "model_type": "regressor",
        "model_file": str(model_path),
        "model_hyperparameters": _json_safe(hyperparams),
        "metrics": _json_safe(metrics),
        "performance_file": str(performance_path),
        "policies": {
            "include_noisy": include_noisy,
            "include_noisy_periodic": include_noisy_periodic,
            "include_main_periodic": include_main_periodic,
        },
        "data": {
            "main_simulation_files": [str(path) for path in main_paths],
            "noisy_simulation_files": [str(path) for path in noisy_paths],
            "main_simulation_configs": _json_safe(main_configs),
            "train_sequences": len(train_sequences),
            "evaluation_sequences": len(eval_sequences),
            "noisy_train_sequences": len(noisy_train_sequences),
            "noisy_eval_folders": len(noisy_eval_groups),
            "train_windows": int(X_train.shape[0]),
            "evaluation_windows": int(X_eval.shape[0]),
        },
        "visualization_files": {key: str(value) for key, value in artifacts.get("figures", {}).items()},
        "noisy_evaluation": _json_safe(noisy_results),
    }

    model_config_path = model_dir / "config.json"
    with open(model_config_path, "w") as f:
        json.dump(model_config_dump, f, indent=2)

    return {
        "model_dir": model_dir,
        "model_path": model_path,
        "config_path": model_config_path,
        "performance_path": performance_path,
        "metrics": metrics,
        "figures": artifacts.get("figures", {}),
        "noisy_evaluation": noisy_results,
        "config": model_config_dump,
    }


def train_forecaster_random_stim_eval(
    source,
    model=None,
    output_root=None,
    visualization=False,
    include_repetitive_eval=True,
    random_state=None,
    **simulation_kwargs,
):
    """
    Train a forecaster on random-stim cells and evaluate on random holdout plus
    optional repetitive-stim cells.

    `source` can be one simulation.pkl path, multiple simulation.pkl paths, or a
    normal simulation config/root-folder input. Non-pkl sources trigger a fresh
    standard simulation before forecaster training.
    """
    from aisam.comptools.regression_forecaster import (
        RegressionForecaster,
        make_window_dataset,
        regression_metrics,
        split_sequences_by_cell,
    )

    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_config = _resolve_model_config(model or {"type": "regressor"})
    model_type = str(model_config["type"]).lower()
    if model_type not in {"regressor", "regression", "regression_forecaster"}:
        raise NotImplementedError(f"Model type {model_config['type']!r} is not implemented yet.")

    if _is_simulation_path_source(source):
        paths = _coerce_simulation_paths(source)
        random_sequences, repetitive_sequences, run_configs = _sequences_by_stim_group_from_paths(
            paths,
            model_config=model_config,
        )
        default_model_root = paths[0].parent / "models"
        simulation_result = None
    else:
        if not include_repetitive_eval:
            simulation_kwargs.setdefault("include_repetitive_stims", False)
        simulation_result = run_training_simulation(source, include_cells=True, **simulation_kwargs)
        random_sequences, repetitive_sequences = _sequences_by_stim_group_from_cells(
            simulation_result["cells"],
            run_config=simulation_result["config"],
            run_prefix="standard",
            model_config=model_config,
        )
        run_configs = [simulation_result["config"]]
        default_model_root = Path(simulation_result["simulation_path"]).parent / "models"
        simulation_result.pop("cells", None)

    if not random_sequences:
        raise ValueError("No random-stim sequences were found for forecaster training.")
    if include_repetitive_eval and not repetitive_sequences:
        raise ValueError("No repetitive-stim sequences were found for forecaster evaluation.")

    hyperparams = _forecaster_hyperparams_from_model_config(model_config)
    hyperparams.setdefault("past_feature_window", 20)
    hyperparams.setdefault("future_window", 1)
    hyperparams.setdefault("output_species", "F")
    if random_state is not None:
        hyperparams.setdefault("random_state", random_state)
    validation_fraction = float(hyperparams.get("validation_fraction", 0.2))
    if validation_fraction <= 0 or validation_fraction >= 1:
        raise ValueError("validation_fraction must be in the range (0, 1) for random-stim holdout evaluation.")

    train_random_sequences, holdout_random_sequences = split_sequences_by_cell(
        random_sequences,
        validation_fraction=validation_fraction,
        random_state=hyperparams.get("random_state"),
    )
    evaluation_sequences = list(holdout_random_sequences)
    if include_repetitive_eval:
        evaluation_sequences.extend(repetitive_sequences)
    window_args = _window_args_from_hyperparams(hyperparams)
    X_train, y_train, train_meta = make_window_dataset(train_random_sequences, **window_args)
    X_eval, y_eval, eval_meta = make_window_dataset(evaluation_sequences, **window_args)

    forecaster = RegressionForecaster(
        past_feature_window=window_args["past_feature_window"],
        future_window=window_args["future_window"],
        past_input_window=window_args["past_input_window"],
        future_input_window=window_args["future_input_window"],
        regressor=hyperparams.get("regressor"),
        normalize=hyperparams.get("normalize", True),
    )
    forecaster.fit(X_train, y_train)
    eval_predictions = forecaster.predict(X_eval)
    metrics = {
        "train_random_stims": forecaster.evaluate(X_train, y_train),
        "random_holdout_plus_repetitive_evaluation"
        if include_repetitive_eval
        else "random_holdout_evaluation": regression_metrics(y_eval, eval_predictions),
    }
    first_config = next((cfg for cfg in run_configs if cfg), {})
    _assign_saved_sampling_metadata(forecaster, first_config)

    dataset = {
        "X_train": X_train,
        "y_train": y_train,
        "train_meta": train_meta,
        "train_sequences": train_random_sequences,
        "validation_sequences": evaluation_sequences,
        "X_validation": X_eval,
        "y_validation": y_eval,
        "validation_meta": eval_meta,
        "validation_predictions": eval_predictions,
    }

    model_root = Path(output_root or model_config.get("output_root") or default_model_root)
    model_dir = model_root / f"random_stim_regressor_{run_stamp}"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.pkl"
    with open(model_path, "wb") as f:
        dill.dump(forecaster, f)

    artifacts = _save_evaluation_artifacts(
        forecaster=forecaster,
        dataset=dataset,
        output_dir=model_dir / "figures",
        output_species=hyperparams.get("output_species", "F"),
        random_state=random_state,
        title="evaluation",
    )
    performance_path = _save_performance_info(
        model_dir / "performance.json",
        metrics=metrics,
        dataset=dataset,
        label="evaluation",
    )

    model_config_dump = {
        "created_at": run_stamp,
        "model_type": "random_stim_regressor",
        "model_file": str(model_path),
        "model_hyperparameters": _json_safe(hyperparams),
        "metrics": _json_safe(metrics),
        "performance_file": str(performance_path),
        "data": {
            "training_policy": "random_stim_cells_train_split_only",
            "evaluation_policy": "random_stim_holdout_plus_all_repetitive_stim_cells"
            if include_repetitive_eval
            else "random_stim_holdout_only",
            "include_repetitive_eval": include_repetitive_eval,
            "simulation_configs": _json_safe(run_configs),
            "random_train_sequences": len(train_random_sequences),
            "random_holdout_sequences": len(holdout_random_sequences),
            "available_repetitive_sequences": len(repetitive_sequences),
            "repetitive_eval_sequences": len(repetitive_sequences) if include_repetitive_eval else 0,
            "total_evaluation_sequences": len(evaluation_sequences),
            "train_windows": int(X_train.shape[0]),
            "evaluation_windows": int(X_eval.shape[0]),
        },
        "simulation_result": _json_safe(simulation_result) if simulation_result else None,
        "visualization_files": {key: str(value) for key, value in artifacts.get("figures", {}).items()},
    }

    model_config_path = model_dir / "config.json"
    with open(model_config_path, "w") as f:
        json.dump(model_config_dump, f, indent=2)

    return {
        "model_dir": model_dir,
        "model_path": model_path,
        "config_path": model_config_path,
        "performance_path": performance_path,
        "metrics": metrics,
        "figures": artifacts.get("figures", {}),
        "config": model_config_dump,
    }


def _resolve_model_config(model):
    if model is None or model is False:
        return None
    if isinstance(model, (str, PathLike)):
        path = Path(model)
        if path.suffix == ".json" and path.exists():
            with open(path, "r") as f:
                loaded = json.load(f)
            return _resolve_model_config(loaded)
        return {"type": str(model)}
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

    hyperparams = _forecaster_hyperparams_from_model_config(model_config)

    hyperparams.setdefault("past_feature_window", 20)
    hyperparams.setdefault("future_window", 1)
    hyperparams.setdefault("output_species", "F")

    forecaster, metrics, dataset = train_regression_forecaster(cells, **hyperparams)
    _assign_saved_sampling_metadata(forecaster, data_config)

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

    artifacts = {}
    if "X_validation" in dataset:
        artifacts = _save_evaluation_artifacts(
            forecaster=forecaster,
            dataset=dataset,
            output_dir=model_dir / "figures",
            output_species=hyperparams.get("output_species", "F"),
            random_state=hyperparams.get("random_state"),
            title="validation",
        )
    performance_path = _save_performance_info(
        model_dir / "performance.json",
        metrics=metrics,
        dataset=dataset if "X_validation" in dataset else None,
        label="validation",
    )

    model_config_dump = {
        "created_at": run_stamp,
        "model_type": "regressor",
        "model_file": str(model_path),
        "model_hyperparameters": _json_safe(hyperparams),
        "metrics": _json_safe(metrics),
        "performance_file": str(performance_path),
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
        "visualization_files": {key: str(value) for key, value in artifacts.get("figures", {}).items()},
    }

    model_config_path = model_dir / "config.json"
    with open(model_config_path, "w") as f:
        json.dump(model_config_dump, f, indent=2)

    return {
        "model_dir": model_dir,
        "model_path": model_path,
        "config_path": model_config_path,
        "performance_path": performance_path,
        "metrics": metrics,
        "figures": artifacts.get("figures", {}),
        "config": model_config_dump,
    }


def _forecaster_hyperparams_from_model_config(model_config):
    hyperparams = dict(model_config.get("hyperparameters", model_config.get("hyperparams", {})))
    excluded = {
        "type",
        "model",
        "name",
        "model_type",
        "hyperparameters",
        "hyperparams",
        "output_root",
        "visualization",
        "sampling",
        "sample_interval_minutes",
        "training_sample_interval_minutes",
        "forecaster_sample_interval_minutes",
        "sample_rate_minutes",
        "interval_rate",
        "data",
        "train_random_stims_only",
        "include_periodic_stims_in_validation",
        "include_main_periodic",
        "include_noisy",
        "include_noisy_periodic",
    }
    for key, value in model_config.items():
        if key not in excluded:
            hyperparams.setdefault(key, value)
    return hyperparams


def _coerce_simulation_paths(simulation_paths):
    if isinstance(simulation_paths, (str, PathLike)):
        paths = [Path(simulation_paths)]
    else:
        paths = [Path(path) for path in simulation_paths]
    if not paths:
        raise ValueError("Provide at least one simulation.pkl path.")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Simulation files not found: {missing}")
    return paths


def _resolve_main_simulation_paths(path):
    if path is None:
        path = Path.cwd()
    if isinstance(path, (str, PathLike)):
        path = Path(path)
        if path.is_dir():
            path = path / "simulation.pkl"
        return _coerce_simulation_paths(path)
    resolved = []
    for item in path:
        item_path = Path(item)
        resolved.append(item_path / "simulation.pkl" if item_path.is_dir() else item_path)
    return _coerce_simulation_paths(resolved)


def _discover_noisy_simulation_paths(main_simulation_path):
    root = Path(main_simulation_path).parent
    noisy_root = root / "noisy"
    if not noisy_root.exists():
        return []
    paths = sorted(noisy_root.glob("sim_*/simulation.pkl"))
    return [path for path in paths if path.exists()]


def _load_and_merge_cells(paths):
    merged = {}
    for run_index, path in enumerate(paths):
        with open(path, "rb") as f:
            cells = dill.load(f)
        for cell_id, cell in cells.items():
            merged[f"run{run_index + 1}_cell{cell_id}"] = cell
    return merged


def _load_run_config_from_simulation_path(path):
    config_path = Path(path).parent / "config.json"
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return json.load(f)


def _sampling_from_run_config(config):
    simulated_cells = config.get("simulated_cells", {})
    if "sampling" in simulated_cells:
        return simulated_cells["sampling"]
    user_config = config.get("user_config", {})
    if "sampling" in user_config:
        return user_config["sampling"]
    return None


def _is_simulation_path_source(source):
    if isinstance(source, (str, PathLike)):
        return Path(source).suffix == ".pkl"
    if isinstance(source, (list, tuple)):
        return all(Path(path).suffix == ".pkl" for path in source)
    return False


def _sequences_by_stim_group_from_paths(paths, model_config, run_label="run"):
    random_sequences = []
    repetitive_sequences = []
    run_configs = []

    for run_index, path in enumerate(paths):
        with open(path, "rb") as f:
            cells = dill.load(f)
        run_config = _load_run_config_from_simulation_path(path)
        run_configs.append(run_config)
        run_random, run_repetitive = _sequences_by_stim_group_from_cells(
            cells,
            run_config=run_config,
            run_prefix=f"{run_label}{run_index + 1}",
            model_config=model_config,
        )
        random_sequences.extend(run_random)
        repetitive_sequences.extend(run_repetitive)

    return random_sequences, repetitive_sequences, run_configs


def _sequences_by_stim_group_from_cells(cells, run_config, run_prefix, model_config):
    from aisam.comptools.regression_forecaster import cells_to_sequences

    hyperparams = _forecaster_hyperparams_from_model_config(model_config)
    feature_species = hyperparams.get("feature_species")
    output_species = hyperparams.get("output_species", "F")
    random_range, red_range, green_range = _stim_ranges_from_run_config(run_config, cells)

    random_cells = {}
    repetitive_cells = {}
    for cell_id, cell in cells.items():
        numeric_id = _numeric_cell_id(cell_id)
        prefixed_id = f"{run_prefix}_cell{cell_id}"
        if _id_in_range(numeric_id, random_range):
            random_cells[prefixed_id] = cell
        elif _id_in_range(numeric_id, red_range) or _id_in_range(numeric_id, green_range):
            repetitive_cells[prefixed_id] = cell

    random_sequences = cells_to_sequences(
        random_cells,
        feature_species=feature_species,
        output_species=output_species,
    )
    repetitive_sequences = cells_to_sequences(
        repetitive_cells,
        feature_species=feature_species,
        output_species=output_species,
    )
    return random_sequences, repetitive_sequences


def _stim_ranges_from_run_config(run_config, cells):
    simulated_cells = run_config.get("simulated_cells", {})
    total_cells = simulated_cells.get("total_cells")
    if total_cells is None:
        total_cells = len(cells)
    fallback = stimulation_cell_ranges(total_cells)
    return (
        simulated_cells.get("random_stimulation_cells", fallback["random_stimulation_cells"]),
        simulated_cells.get(
            "repetitive_stimulation_cells_red_first",
            fallback["repetitive_stimulation_cells_red_first"],
        ),
        simulated_cells.get(
            "repetitive_stimulation_cells_green_first",
            fallback["repetitive_stimulation_cells_green_first"],
        ),
    )


def _numeric_cell_id(cell_id):
    try:
        return int(cell_id)
    except (TypeError, ValueError):
        digits = "".join(ch for ch in str(cell_id) if ch.isdigit())
        if not digits:
            raise ValueError(f"Cannot infer numeric cell id from {cell_id!r}.")
        return int(digits)


def _id_in_range(cell_id, cell_range):
    if cell_range is None:
        return False
    if len(cell_range) != 2:
        return False
    start, end = cell_range
    return int(start) <= int(cell_id) <= int(end)


def _window_args_from_hyperparams(hyperparams):
    return {
        "past_feature_window": hyperparams["past_feature_window"],
        "future_window": hyperparams["future_window"],
        "past_input_window": hyperparams.get("past_input_window"),
        "future_input_window": hyperparams.get("future_input_window"),
        "stride": hyperparams.get("stride", 1),
    }


def _assign_saved_sampling_metadata(forecaster, run_config):
    simulated_cells = run_config.get("simulated_cells", {})
    sample_interval = simulated_cells.get("sample_interval_minutes")
    if sample_interval is None:
        sample_interval = _sample_interval_from_config(run_config.get("user_config", {}))
    forecaster.sample_interval_minutes = sample_interval
    forecaster.sampling = simulated_cells.get("saved_sampling", simulated_cells.get("sampling"))


def _evaluate_sequences(forecaster, sequences, hyperparams):
    from aisam.comptools.regression_forecaster import make_window_dataset, regression_metrics

    window_args = _window_args_from_hyperparams(hyperparams)
    X_eval, y_eval, eval_meta = make_window_dataset(sequences, **window_args)
    predictions = forecaster.predict(X_eval)
    metrics = regression_metrics(y_eval, predictions)
    dataset = {
        "validation_sequences": sequences,
        "X_validation": X_eval,
        "y_validation": y_eval,
        "validation_meta": eval_meta,
        "validation_predictions": predictions,
    }
    return dataset, metrics


def _save_evaluation_artifacts(
    forecaster,
    dataset,
    output_dir,
    output_species="F",
    random_state=None,
    title="evaluation",
):
    from aisam.utils.visualization_tools import (
        plot_error_distribution,
        plot_forecaster_evaluation_examples,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = plot_forecaster_evaluation_examples(
        forecaster=forecaster,
        dataset=dataset,
        output_dir=output_dir,
        output_species=output_species,
        random_state=random_state,
    )
    error_info = _window_error_info(dataset["y_validation"], dataset["validation_predictions"])
    histogram_path = plot_error_distribution(
        error_info["per_window_rmse"],
        output_dir / "error_distribution.svg",
        title=f"{title} error distribution",
    )
    figures["error_distribution"] = histogram_path
    return {"figures": figures, "error_info": error_info}


def _save_performance_info(path, metrics, dataset=None, label="evaluation"):
    info = {
        "label": label,
        "metrics": _json_safe(metrics),
    }
    if dataset is not None:
        info["error_distribution"] = _json_safe(
            _window_error_info(dataset["y_validation"], dataset["validation_predictions"])
        )
        info["num_windows"] = int(np.asarray(dataset["y_validation"]).shape[0])
        info["num_sequences"] = len(dataset.get("validation_sequences", []))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    return path


def _window_error_info(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred
    per_window_mse = np.mean(residual ** 2, axis=1)
    per_window_rmse = np.sqrt(per_window_mse)
    per_window_mae = np.mean(np.abs(residual), axis=1)
    return {
        "mean_rmse": float(np.mean(per_window_rmse)),
        "median_rmse": float(np.median(per_window_rmse)),
        "std_rmse": float(np.std(per_window_rmse)),
        "min_rmse": float(np.min(per_window_rmse)),
        "max_rmse": float(np.max(per_window_rmse)),
        "mean_mae": float(np.mean(per_window_mae)),
        "median_mae": float(np.median(per_window_mae)),
        "std_mae": float(np.std(per_window_mae)),
        "per_window_rmse": per_window_rmse.tolist(),
    }


def _normalize_policy(value):
    if value is None or value is False:
        return "none"
    if value is True:
        return "eval"
    value = str(value).strip().lower()
    aliases = {
        "ignore": "none",
        "off": "none",
        "false": "none",
        "0": "none",
        "validation": "eval",
        "evaluate": "eval",
        "true": "eval",
        "1": "eval",
    }
    value = aliases.get(value, value)
    if value not in {"train", "eval", "none"}:
        raise ValueError("Policy values must be one of: train, eval, none.")
    return value


def _policy_enabled(value):
    return _normalize_policy(value) != "none"


def _model_label(model_config, include_noisy, include_main_periodic):
    model_type = str(model_config.get("type", "regressor")).lower()
    parts = []
    if include_main_periodic == "train":
        parts.append("periodic")
    else:
        parts.append("random")
    if include_noisy == "train":
        parts.append("with_noisy")
    elif include_noisy == "eval":
        parts.append("noisy_eval")
    parts.append(model_type)
    return "_".join(parts)
