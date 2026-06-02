import json
from datetime import datetime
from os import PathLike
from pathlib import Path

import dill
import numpy as np

from aisam.model import sims
from aisam.utils import aux


def run_sanity_check_simulation(config, species=None, num_realizations=None, progress=True):
    """
    Run a 30-cell sanity-check simulation and save random/periodic panels.

    The panel contains 10 random-stim cells, 10 green-first periodic cells with
    num_repeat 1..10, and 10 red-first periodic cells with num_repeat 1..10.
    """
    from aisam.utils.visualization_tools import plot_sanity_simulation_group

    config, source_root, _ = _load_simulation_config_from_root(config)
    params = _load_params_from_config(config, source_root=source_root)
    label = config.get("label", config.get("circuit", "CcaSR"))
    t_max = int(config.get("t_max", params.get("t_max")))
    sampling = int(config.get("sampling", params.get("sampling", 10)))
    sample_interval = _sample_interval_from_config(config)
    stim_points = _stim_points_for_interval(t_max, sample_interval)
    params.setdefault("t_max", t_max)
    params.setdefault("sampling", sampling)
    num_realizations = int(num_realizations or config.get("num_realizations", 1))
    species = species or config.get("sanity_species", ["F"])
    species = [species] if isinstance(species, str) else list(species)

    stims = {}
    for i in range(1, 11):
        stims[f"cell {i}"] = aux.random_stim_maker(stim_points).tolist()
    for repeat in range(1, 11):
        stims[f"cell {10 + repeat}"] = aux.repetitive_stim_maker(
            num_repeat=repeat,
            total_time=stim_points,
            off_first=False,
        ).tolist()
        stims[f"cell {20 + repeat}"] = aux.repetitive_stim_maker(
            num_repeat=repeat,
            total_time=stim_points,
            off_first=True,
        ).tolist()

    xpt = sims.experiment(params, label)
    xpt.init_exp(30)
    xpt.run_training_sim(stims, num_realizations, progress=progress, desc=f"{label} sanity check")
    saved_sampling = _sample_cells_at_interval(
        xpt.Cells,
        simulation_sampling=sampling,
        interval_minutes=sample_interval,
    )

    root = Path(config.get("root_folder", source_root or dated_assets_root()))
    output_dir = root / "sanity_plots"
    figure_paths = {
        "random_stims": plot_sanity_simulation_group(
            xpt.Cells,
            cell_ids=range(1, 11),
            species=species,
            save_path=output_dir / "random_stims.svg",
            title="random_stims",
            sampling=saved_sampling,
        ),
        "periodic_green_first": plot_sanity_simulation_group(
            xpt.Cells,
            cell_ids=range(11, 21),
            species=species,
            save_path=output_dir / "periodic_green_first.svg",
            title="periodic_green_first",
            sampling=saved_sampling,
        ),
        "periodic_red_first": plot_sanity_simulation_group(
            xpt.Cells,
            cell_ids=range(21, 31),
            species=species,
            save_path=output_dir / "periodic_red_first.svg",
            title="periodic_red_first",
            sampling=saved_sampling,
        ),
    }

    return {
        "output_dir": output_dir,
        "figures": figure_paths,
        "cells": xpt.Cells,
        "stims": stims,
        "saved_sampling": saved_sampling,
        "sample_interval_minutes": sample_interval,
    }


def run_training_simulation(
    root_folder,
    label=None,
    total_cell=None,
    noisy_sims=None,
    noisy_total_cells=None,
    temperatures=None,
    output_root=None,
    random_seed=None,
    include_repetitive_stims=None,
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
    if include_repetitive_stims is not None:
        config["include_repetitive_stims"] = include_repetitive_stims
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
    sample_interval = _sample_interval_from_config(config)
    include_repetitive_stims = bool(config.get("include_repetitive_stims", True))
    params.setdefault("t_max", t_max)
    params.setdefault("sampling", sampling)

    random_seed = config.get("random_seed")
    if random_seed is not None:
        np.random.seed(int(random_seed))

    stims = build_standard_stims(
        t_max,
        total_cells=total_cells,
        sample_interval_minutes=sample_interval,
        include_repetitive_stims=include_repetitive_stims,
    )
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
        sample_interval_minutes=sample_interval,
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
        noisy_stims = build_standard_stims(
            t_max,
            total_cells=noisy_total_cells,
            sample_interval_minutes=sample_interval,
            include_repetitive_stims=include_repetitive_stims,
        )
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
                sample_interval_minutes=sample_interval,
            )
            noisy_result.pop("cells", None)
            result["noisy"].append(noisy_result)

    return result


def build_standard_stims(
    total_time,
    total_cells=1000,
    sample_interval_minutes=None,
    include_repetitive_stims=True,
):
    total_cells = _validate_total_cells(total_cells)
    random_cells = total_cells - 100 if include_repetitive_stims else total_cells
    red_first_start = random_cells + 1
    green_first_start = random_cells + 51
    stim_points = _stim_points_for_interval(total_time, sample_interval_minutes)

    stims = {}
    for i in range(1, random_cells + 1):
        stims[f"cell {i}"] = aux.random_stim_maker(stim_points).tolist()
    if include_repetitive_stims:
        for i in range(red_first_start, green_first_start):
            stims[f"cell {i}"] = aux.repetitive_stim_maker(
                num_repeat=i - random_cells,
                total_time=stim_points,
                off_first=True,
            ).tolist()
        for i in range(green_first_start, total_cells + 1):
            stims[f"cell {i}"] = aux.repetitive_stim_maker(
                num_repeat=i - (random_cells + 50),
                total_time=stim_points,
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


def stimulation_cell_ranges(total_cells, include_repetitive_stims=True):
    total_cells = _validate_total_cells(total_cells)
    random_cells = total_cells - 100 if include_repetitive_stims else total_cells
    red_range = [random_cells + 1, random_cells + 50] if include_repetitive_stims else []
    green_range = [random_cells + 51, total_cells] if include_repetitive_stims else []
    return {
        "total_cells": total_cells,
        "random_stimulation_cells": [1, random_cells],
        "repetitive_stimulation_cells_red_first": red_range,
        "repetitive_stimulation_cells_green_first": green_range,
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


def _sample_cells_at_interval(cells, simulation_sampling, interval_minutes):
    if interval_minutes is None:
        return simulation_sampling

    step = max(1, int(round(float(simulation_sampling) * float(interval_minutes))))
    saved_sampling = 1.0 / float(interval_minutes)

    for cell in cells.values():
        for run_index, stim_vec in enumerate(cell.stims):
            trace_length = _cell_run_trace_length(cell, run_index)
            indices = np.arange(0, trace_length, step)
            dense_input = _expand_stim_to_trace(stim_vec, trace_length)
            cell.stims[run_index] = dense_input[indices].reshape(-1).tolist()

            for species in cell.expressions:
                trace = np.asarray(cell.expressions[species][run_index])
                if trace.ndim == 1:
                    sampled_trace = trace[indices]
                elif trace.ndim == 2:
                    sampled_trace = trace[:, indices]
                else:
                    raise ValueError(
                        f"Expected 1D or 2D expression trace for {species!r}, got shape {trace.shape}."
                    )
                cell.expressions[species][run_index] = sampled_trace

    return saved_sampling


def _stim_points_for_interval(t_max, interval_minutes):
    if interval_minutes is None:
        return int(t_max)
    return max(1, int(float(t_max) / float(interval_minutes)))


def _cell_run_trace_length(cell, run_index):
    for species in cell.expressions:
        trace = np.asarray(cell.expressions[species][run_index])
        if trace.ndim == 1:
            return trace.shape[0]
        if trace.ndim == 2:
            return trace.shape[1]
    raise ValueError("Cell has no expression traces to sample.")


def _expand_stim_to_trace(stim_vec, trace_length):
    stim = np.asarray(stim_vec, dtype=float).reshape(-1, 1)
    repeats = int(np.ceil(trace_length / len(stim)))
    expanded = np.repeat(stim, repeats, axis=0)
    return expanded[:trace_length]


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
    sample_interval_minutes=None,
):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    xpt = sims.experiment(params, label)
    xpt.init_exp(num_cells)
    xpt.run_training_sim(stims, num_realizations, progress=progress, desc=desc)
    saved_sampling = _sample_cells_at_interval(
        xpt.Cells,
        simulation_sampling=sampling,
        interval_minutes=sample_interval_minutes,
    )

    save_parquet = bool(user_config.get("save_parquet", True))
    save_pickle = bool(user_config.get("save_pickle", True))
    if not save_parquet and not save_pickle:
        raise ValueError("At least one of save_parquet or save_pickle must be enabled.")

    parquet_path = None
    if save_parquet:
        parquet_path = run_dir / "simulation.parquet"
        time_step_minutes = (
            sample_interval_minutes
            if sample_interval_minutes is not None
            else 1.0 / float(saved_sampling)
        )
        save_simulation_dataframe(
            xpt.Cells,
            parquet_path,
            time_step_minutes=time_step_minutes,
        )

    pickle_path = None
    if save_pickle:
        pickle_path = run_dir / "simulation.pkl"
        with open(pickle_path, "wb") as f:
            dill.dump(xpt.Cells, f)

    simulation_path = parquet_path or pickle_path

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
        "simulation_data_file": str(simulation_path),
        "simulation_parquet_file": str(parquet_path) if parquet_path is not None else None,
        "simulation_pickle_file": str(pickle_path) if pickle_path is not None else None,
        "simulation_params_file": str(params_path),
        "stims_file": str(stims_path),
        "data_format": "parquet_long_v1" if parquet_path is not None else "cell_pickle_v1",
        "data_columns": ["cell_id", "realization", "species", "stim", "time", "value"],
        "simulated_cells": {
            **stimulation_cell_ranges(
                num_cells,
                include_repetitive_stims=bool(user_config.get("include_repetitive_stims", True)),
            ),
            "include_repetitive_stims": bool(user_config.get("include_repetitive_stims", True)),
            "num_realizations": num_realizations,
            "t_max": t_max,
            "sampling": sampling,
            "saved_sampling": saved_sampling,
            "sample_interval_minutes": sample_interval_minutes,
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
        "data_path": simulation_path,
        "parquet_path": parquet_path,
        "simulation_pickle_path": pickle_path,
        "params_path": params_path,
        "stims_path": stims_path,
        "config_path": config_path,
        "config": config_dump,
        "cells": xpt.Cells,
    }


def save_simulation_dataframe(cells, path, time_step_minutes=5, value_column="value"):
    """
    Save Cell_sim outputs as a long-form parquet table.

    The primary schema is:
    cell_id, realization, species, stim, time, value
    """
    dataframe = cells_to_simulation_dataframe(
        cells,
        time_step_minutes=time_step_minutes,
        value_column=value_column,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(path, index=False)
    return path


def cells_to_simulation_dataframe(cells, time_step_minutes=5, value_column="value"):
    """
    Convert saved Cell_sim objects into a long-form pandas dataframe.

    Each row is one cell/realization/species/time observation. `stim` is the
    stimulation value active at that time point.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Saving simulation parquet files requires pandas. Install pandas "
            "and pyarrow, or set save_parquet=false in the run config."
        ) from exc

    frames = []
    for cell_id in _sorted_cell_ids(cells):
        cell = cells[cell_id]
        for run_index, stim_vec in enumerate(cell.stims):
            trace_length = _cell_run_trace_length(cell, run_index)
            stim_trace = _expand_stim_to_trace(stim_vec, trace_length).reshape(-1)
            time_values = _time_values(trace_length, time_step_minutes)
            num_realizations = _cell_num_realizations(cell, run_index)

            for species, runs in cell.expressions.items():
                species_runs = _as_trace_matrix(runs[run_index])
                if species_runs.shape[1] != trace_length:
                    raise ValueError(
                        f"Species {species!r} in cell {cell_id!r} has trace length "
                        f"{species_runs.shape[1]}, expected {trace_length}."
                    )

                for realization_index in range(num_realizations):
                    source_index = 0 if species_runs.shape[0] == 1 else realization_index
                    frames.append(
                        pd.DataFrame(
                            {
                                "cell_id": _table_cell_id(cell_id),
                                "realization": realization_index + 1,
                                "species": species,
                                "stim": stim_trace,
                                "time": time_values,
                                value_column: species_runs[source_index].astype(float),
                            }
                        )
                    )

    if not frames:
        return pd.DataFrame(
            columns=["cell_id", "realization", "species", "stim", "time", value_column]
        )
    dataframe = pd.concat(frames, ignore_index=True)
    return dataframe[["cell_id", "realization", "species", "stim", "time", value_column]]


def _cell_num_realizations(cell, run_index):
    num_realizations = 1
    for runs in cell.expressions.values():
        values = _as_trace_matrix(runs[run_index])
        num_realizations = max(num_realizations, values.shape[0])
    return num_realizations


def _as_trace_matrix(values):
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim == 2:
        return array
    raise ValueError(f"Expected a 1D or 2D expression trace, got shape {array.shape}.")


def _time_values(trace_length, time_step_minutes):
    if time_step_minutes is None:
        time_step_minutes = 1
    values = np.arange(trace_length, dtype=float) * float(time_step_minutes)
    rounded = np.round(values)
    if np.allclose(values, rounded):
        return rounded.astype(int)
    return values


def _table_cell_id(cell_id):
    try:
        return int(cell_id)
    except (TypeError, ValueError):
        return str(cell_id)


def _sorted_cell_ids(cells):
    def key(cell_id):
        try:
            return int(cell_id)
        except (TypeError, ValueError):
            return str(cell_id)

    return sorted(cells.keys(), key=key)


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
