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


def train_forecaster(
    path=None,
    config=None,
    output_root=None,
    visualization=None,
    random_state=None,
    include_noisy="none",
    include_noisy_periodic="eval",
    include_main_periodic="eval",
    label=None,
):
    """
    Train a forecaster from saved simulation data with explicit policies.

    Policies are "train", "eval", or "none":
    - include_main_periodic controls main-run repetitive stim cells.
    - include_noisy controls noisy-run random stim cells.
    - include_noisy_periodic controls noisy-run repetitive stim cells.

    `path` may be a run folder, a simulation parquet file, a legacy
    simulation.pkl file, or a list of those files.
    """
    from aisam.comptools.regression_forecaster import make_window_dataset, split_sequences_by_cell

    main_paths = _resolve_main_simulation_paths(path)
    noisy_paths = _discover_noisy_simulation_paths(main_paths[0])
    model_config = _resolve_model_config(config or {"type": "regressor"})
    if model_config is None:
        raise ValueError("A model config is required for forecaster training.")
    model_type = str(model_config["type"]).lower()
    if not _is_supported_model_type(model_type):
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

    forecaster = _build_forecaster(model_type, model_config, hyperparams, window_args, train_sequences)
    forecaster.fit(X_train, y_train)
    eval_predictions = forecaster.predict(X_eval)
    metrics = {
        "train": forecaster.evaluate(X_train, y_train),
        "evaluation": _regression_metrics(y_eval, eval_predictions),
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

    artifacts = {}
    if visualization:
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
        noisy_artifacts = {}
        if visualization:
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
        "model_type": model_type,
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


def train_forecaster_from_simulation(
    simulation_paths,
    model=None,
    output_root=None,
    visualization=None,
    random_state=None,
    include_noisy="none",
    include_noisy_periodic="none",
    include_main_periodic="train",
):
    """
    Compatibility wrapper around `train_forecaster`.

    Prefer `train_forecaster` for new code.
    """
    return train_forecaster(
        path=simulation_paths,
        config=model,
        output_root=output_root,
        visualization=visualization,
        random_state=random_state,
        include_noisy=include_noisy,
        include_noisy_periodic=include_noisy_periodic,
        include_main_periodic=include_main_periodic,
    )


def train_forecaster_from_simulation_config(
    path=None,
    config=None,
    output_root=None,
    visualization=None,
    random_state=None,
    include_noisy="none",
    include_noisy_periodic="eval",
    include_main_periodic="eval",
    label=None,
):
    """
    Compatibility wrapper around `train_forecaster`.

    Prefer `train_forecaster` for new code.
    """
    return train_forecaster(
        path=path,
        config=config,
        output_root=output_root,
        visualization=visualization,
        random_state=random_state,
        include_noisy=include_noisy,
        include_noisy_periodic=include_noisy_periodic,
        include_main_periodic=include_main_periodic,
        label=label,
    )


def train_forecaster_random_stim_eval(
    source,
    model=None,
    output_root=None,
    visualization=None,
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
    model_config = _resolve_model_config(model or {"type": "regressor"})
    model_type = str(model_config["type"]).lower()
    if not _is_supported_model_type(model_type):
        raise NotImplementedError(f"Model type {model_config['type']!r} is not implemented yet.")

    if _is_simulation_path_source(source):
        training_path = source
        simulation_result = None
    else:
        if not include_repetitive_eval:
            simulation_kwargs.setdefault("include_repetitive_stims", False)
        simulation_result = run_training_simulation(source, include_cells=False, **simulation_kwargs)
        training_path = simulation_result["simulation_path"]

    result = train_forecaster(
        path=training_path,
        config=model_config,
        output_root=output_root,
        visualization=visualization,
        random_state=random_state,
        include_noisy="none",
        include_noisy_periodic="none",
        include_main_periodic="eval" if include_repetitive_eval else "none",
        label=f"random_stim_{model_type}",
    )
    if simulation_result is not None:
        result["simulation_result"] = simulation_result
    return result


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
    if _is_supported_model_type(model_type):
        data_path = _simulation_data_path_from_config(data_config)
        if data_path is not None:
            return train_forecaster(
                path=data_path,
                config=model_config,
                visualization=bool(model_config.get("visualization", False)),
                random_state=model_config.get("random_state"),
                include_noisy="none",
                include_noisy_periodic="none",
                include_main_periodic="train",
                label=model_type,
            )
        return _train_and_save_regressor(model_config, cells, data_config, run_stamp)
    raise NotImplementedError(f"Model type {model_config['type']!r} is not implemented yet.")


def _train_and_save_regressor(model_config, cells, data_config, run_stamp):
    hyperparams = _forecaster_hyperparams_from_model_config(model_config)

    hyperparams.setdefault("past_feature_window", 20)
    hyperparams.setdefault("future_window", 1)
    hyperparams.setdefault("output_species", "F")

    forecaster, metrics, dataset = _train_model_from_cells(cells, model_config, hyperparams)
    _assign_saved_sampling_metadata(forecaster, data_config)

    model_root = Path(
        model_config.get(
            "output_root",
            Path(data_config["simulation_file"]).parent / "models",
        )
    )
    model_type = str(model_config["type"]).lower()
    model_dir = model_root / f"{model_type}_{run_stamp}"
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
        "model_type": str(model_config["type"]).lower(),
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


def _is_supported_model_type(model_type):
    return str(model_type).lower() in {
        "regressor",
        "regression",
        "regression_forecaster",
        "lstm",
        "lstm_encoder_decoder",
        "lstm_encoder_decoder_forecaster",
    }


def _is_lstm_model_type(model_type):
    return str(model_type).lower() in {
        "lstm",
        "lstm_encoder_decoder",
        "lstm_encoder_decoder_forecaster",
    }


def _train_model_from_cells(cells, model_config, hyperparams):
    from aisam.comptools.regression_forecaster import (
        cells_to_sequences,
        make_window_dataset,
        split_sequences_by_cell,
    )

    feature_species = hyperparams.get("feature_species")
    output_species = hyperparams.get("output_species", "F")
    sequences = cells_to_sequences(
        cells,
        feature_species=feature_species,
        output_species=output_species,
    )
    train_sequences, validation_sequences = split_sequences_by_cell(
        sequences,
        validation_fraction=hyperparams.get("validation_fraction", 0.2),
        random_state=hyperparams.get("random_state"),
    )
    window_args = _window_args_from_hyperparams(hyperparams)
    X_train, y_train, train_meta = make_window_dataset(train_sequences, **window_args)
    forecaster = _build_forecaster(
        str(model_config["type"]).lower(),
        model_config,
        hyperparams,
        window_args,
        train_sequences,
    )
    forecaster.fit(X_train, y_train)
    metrics = {"train": forecaster.evaluate(X_train, y_train)}
    dataset = {
        "X_train": X_train,
        "y_train": y_train,
        "train_meta": train_meta,
        "train_sequences": train_sequences,
        "validation_sequences": validation_sequences,
    }
    if validation_sequences:
        X_val, y_val, val_meta = make_window_dataset(validation_sequences, **window_args)
        val_predictions = forecaster.predict(X_val)
        metrics["validation"] = _regression_metrics(y_val, val_predictions)
        dataset.update(
            {
                "X_validation": X_val,
                "y_validation": y_val,
                "validation_meta": val_meta,
                "validation_predictions": val_predictions,
            }
        )
    return forecaster, metrics, dataset


def _build_forecaster(model_type, model_config, hyperparams, window_args, train_sequences):
    feature_dim, input_dim = _sequence_dimensions(train_sequences)
    if _is_lstm_model_type(model_type):
        try:
            from aisam.comptools.LSTM_encoder_decoder_forecaster import LSTMEncoderDecoderForecaster
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                raise ImportError("LSTM forecaster training requires PyTorch to be installed.") from exc
            raise

        return LSTMEncoderDecoderForecaster(
            past_feature_window=window_args["past_feature_window"],
            future_window=window_args["future_window"],
            past_input_window=window_args["past_input_window"],
            future_input_window=window_args["future_input_window"],
            feature_dim=feature_dim,
            input_dim=input_dim,
            hidden_size=hyperparams.get("hidden_size", hyperparams.get("hidden_dim", 64)),
            num_layers=hyperparams.get("num_layers", hyperparams.get("layers", 2)),
            dropout=hyperparams.get("dropout", 0.0),
            batch_size=hyperparams.get("batch_size", 32),
            epochs=hyperparams.get("epochs", 10),
            learning_rate=hyperparams.get("learning_rate", hyperparams.get("lr", 0.001)),
            normalize=hyperparams.get("normalize", True),
            device=hyperparams.get("device"),
            random_state=hyperparams.get("random_state"),
            verbose=hyperparams.get("verbose", True),
        )

    from aisam.comptools.regression_forecaster import RegressionForecaster

    return RegressionForecaster(
        past_feature_window=window_args["past_feature_window"],
        future_window=window_args["future_window"],
        past_input_window=window_args["past_input_window"],
        future_input_window=window_args["future_input_window"],
        regressor=hyperparams.get("regressor"),
        normalize=hyperparams.get("normalize", True),
    )


def _sequence_dimensions(sequences):
    if not sequences:
        return 1, 1
    first = sequences[0]
    feature_dim = int(np.asarray(first["features"]).shape[1])
    input_dim = int(np.asarray(first["inputs"]).shape[1])
    return feature_dim, input_dim


def _regression_metrics(y_true, y_pred):
    from aisam.comptools.regression_forecaster import regression_metrics

    return regression_metrics(y_true, y_pred)


def _coerce_simulation_paths(simulation_paths):
    if isinstance(simulation_paths, (str, PathLike)):
        paths = [Path(simulation_paths)]
    else:
        paths = [Path(path) for path in simulation_paths]
    if not paths:
        raise ValueError("Provide at least one simulation data path.")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Simulation data files not found: {missing}")
    return paths


def _resolve_main_simulation_paths(path):
    if path is None:
        path = Path.cwd()
    if isinstance(path, (str, PathLike)):
        path = Path(path)
        if path.is_dir():
            path = _preferred_simulation_file(path)
        return _coerce_simulation_paths(path)
    resolved = []
    for item in path:
        item_path = Path(item)
        resolved.append(_preferred_simulation_file(item_path) if item_path.is_dir() else item_path)
    return _coerce_simulation_paths(resolved)


def _discover_noisy_simulation_paths(main_simulation_path):
    root = Path(main_simulation_path).parent
    noisy_root = root / "noisy"
    if not noisy_root.exists():
        return []
    paths = []
    for sim_dir in sorted(noisy_root.glob("sim_*")):
        if sim_dir.is_dir():
            paths.append(_preferred_simulation_file(sim_dir, required=False))
    return [path for path in paths if path is not None and path.exists()]


def _preferred_simulation_file(run_dir, required=True):
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            run_config = json.load(f)
        for key in ("simulation_data_file", "simulation_parquet_file"):
            configured = Path(run_config[key]) if run_config.get(key) else None
            if configured is None:
                continue
            configured = configured if configured.is_absolute() else run_dir / configured
            if configured.exists():
                return configured

    for filename in ("simulation.parquet", "simulation.pkl"):
        candidate = run_dir / filename
        if candidate.exists():
            return candidate
    if config_path.exists():
        for key in ("simulation_file", "simulation_pickle_file"):
            configured = Path(run_config[key]) if run_config.get(key) else None
            if configured is None:
                continue
            configured = configured if configured.is_absolute() else run_dir / configured
            if configured.exists():
                return configured
    if required:
        raise FileNotFoundError(f"No simulation.parquet or simulation.pkl found in {run_dir}.")
    return None


def _simulation_data_path_from_config(config):
    for key in (
        "simulation_data_file",
        "simulation_parquet_file",
        "simulation_file",
        "simulation_pickle_file",
    ):
        value = config.get(key)
        if value:
            return Path(value)
    return None


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
        path = Path(source)
        return path.suffix in {".parquet", ".pkl"} or (
            path.is_dir() and _preferred_simulation_file(path, required=False) is not None
        )
    if isinstance(source, (list, tuple)):
        return all(_is_simulation_path_source(path) for path in source)
    return False


def _sequences_by_stim_group_from_paths(paths, model_config, run_label="run"):
    random_sequences = []
    repetitive_sequences = []
    run_configs = []

    for run_index, path in enumerate(paths):
        run_config = _load_run_config_from_simulation_path(path)
        run_configs.append(run_config)
        sequences = _sequences_from_simulation_path(path, model_config)
        run_random, run_repetitive = _split_sequences_by_stim_group(sequences, run_config)
        random_sequences.extend(_prefix_sequence_cell_ids(run_random, f"{run_label}{run_index + 1}"))
        repetitive_sequences.extend(_prefix_sequence_cell_ids(run_repetitive, f"{run_label}{run_index + 1}"))

    return random_sequences, repetitive_sequences, run_configs


def _sequences_from_simulation_path(path, model_config):
    from aisam.comptools.regression_forecaster import (
        cells_to_sequences,
        dataframe_to_sequences,
        load_simulation_dataframe,
    )

    hyperparams = _forecaster_hyperparams_from_model_config(model_config)
    feature_species = hyperparams.get("feature_species")
    output_species = hyperparams.get("output_species", "F")
    input_dim = int(hyperparams.get("input_dim", 1))

    path = Path(path)
    if path.suffix == ".parquet":
        return dataframe_to_sequences(
            load_simulation_dataframe(path),
            feature_species=feature_species,
            output_species=output_species,
            input_dim=input_dim,
        )
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            cells = dill.load(f)
        return cells_to_sequences(
            cells,
            feature_species=feature_species,
            output_species=output_species,
            input_dim=input_dim,
        )
    raise ValueError(f"Unsupported simulation data file type: {path}")


def _split_sequences_by_stim_group(sequences, run_config):
    random_range, red_range, green_range = _stim_ranges_from_run_config(run_config, sequences)
    random_sequences = []
    repetitive_sequences = []
    for sequence in sequences:
        numeric_id = _numeric_cell_id(sequence.get("cell_id"))
        if _id_in_range(numeric_id, random_range):
            random_sequences.append(sequence)
        elif _id_in_range(numeric_id, red_range) or _id_in_range(numeric_id, green_range):
            repetitive_sequences.append(sequence)
    return random_sequences, repetitive_sequences


def _prefix_sequence_cell_ids(sequences, run_prefix):
    prefixed = []
    for sequence in sequences:
        copied = dict(sequence)
        original_cell_id = copied.get("cell_id")
        copied["source_cell_id"] = original_cell_id
        copied["cell_id"] = f"{run_prefix}_cell{original_cell_id}"
        prefixed.append(copied)
    return prefixed


def _sequences_by_stim_group_from_cells(cells, run_config, run_prefix, model_config):
    from aisam.comptools.regression_forecaster import cells_to_sequences

    hyperparams = _forecaster_hyperparams_from_model_config(model_config)
    feature_species = hyperparams.get("feature_species")
    output_species = hyperparams.get("output_species", "F")
    sequences = cells_to_sequences(
        cells,
        feature_species=feature_species,
        output_species=output_species,
    )
    random_sequences, repetitive_sequences = _split_sequences_by_stim_group(sequences, run_config)
    return (
        _prefix_sequence_cell_ids(random_sequences, run_prefix),
        _prefix_sequence_cell_ids(repetitive_sequences, run_prefix),
    )


def _stim_ranges_from_run_config(run_config, cells):
    simulated_cells = run_config.get("simulated_cells", {})
    total_cells = simulated_cells.get("total_cells")
    if total_cells is None:
        total_cells = _infer_total_cells(cells)
    if all(
        key in simulated_cells
        for key in (
            "random_stimulation_cells",
            "repetitive_stimulation_cells_red_first",
            "repetitive_stimulation_cells_green_first",
        )
    ):
        fallback = {}
    else:
        fallback = stimulation_cell_ranges(total_cells)
    random_range = simulated_cells.get("random_stimulation_cells")
    red_range = simulated_cells.get("repetitive_stimulation_cells_red_first")
    green_range = simulated_cells.get("repetitive_stimulation_cells_green_first")
    if random_range is None:
        random_range = fallback["random_stimulation_cells"]
    if red_range is None:
        red_range = fallback["repetitive_stimulation_cells_red_first"]
    if green_range is None:
        green_range = fallback["repetitive_stimulation_cells_green_first"]
    return random_range, red_range, green_range


def _numeric_cell_id(cell_id):
    try:
        return int(cell_id)
    except (TypeError, ValueError):
        digits = ""
        for ch in reversed(str(cell_id)):
            if ch.isdigit():
                digits = ch + digits
            elif digits:
                break
        if not digits:
            raise ValueError(f"Cannot infer numeric cell id from {cell_id!r}.")
        return int(digits)


def _infer_total_cells(source):
    if isinstance(source, dict):
        return len(source)
    cell_ids = {
        sequence.get("source_cell_id", sequence.get("cell_id"))
        for sequence in source
        if sequence.get("source_cell_id", sequence.get("cell_id")) is not None
    }
    numeric_ids = []
    for cell_id in cell_ids:
        try:
            numeric_ids.append(_numeric_cell_id(cell_id))
        except ValueError:
            pass
    if numeric_ids:
        return max(numeric_ids)
    return len(cell_ids)


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
    log10_rmse = np.log10(np.clip(per_window_rmse, np.finfo(float).tiny, None))
    return {
        "mean_rmse": float(np.mean(per_window_rmse)),
        "median_rmse": float(np.median(per_window_rmse)),
        "std_rmse": float(np.std(per_window_rmse)),
        "min_rmse": float(np.min(per_window_rmse)),
        "max_rmse": float(np.max(per_window_rmse)),
        "mean_log10_rmse": float(np.mean(log10_rmse)),
        "median_log10_rmse": float(np.median(log10_rmse)),
        "std_log10_rmse": float(np.std(log10_rmse)),
        "mean_mae": float(np.mean(per_window_mae)),
        "median_mae": float(np.median(per_window_mae)),
        "std_mae": float(np.std(per_window_mae)),
        "per_window_rmse": per_window_rmse.tolist(),
        "per_window_log10_rmse": log10_rmse.tolist(),
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
