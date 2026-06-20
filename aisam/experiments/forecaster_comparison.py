"""
Forecaster comparison experiments.

This module orchestrates matrix-style cross-testing: train one forecaster per
training dataset, evaluate that trained forecaster on every configured test
dataset, and aggregate mean RMSE into tabular and visual matrix outputs.
"""

from __future__ import annotations

import argparse
import copy
import json
from os import PathLike
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from aisam.utils.forecaster_training import cross_test_forecaster


DEFAULT_FORECASTER_ROOT = "forecasters"
DEFAULT_REPORTER_SPECIES = "F"
DEFAULT_INCLUDE_MAIN_PERIODIC = "none"
DEFAULT_INCLUDE_TEST_PERIODIC = "none"
DEFAULT_PERIODIC_PLOTS_DIR = "periodic_sanity_plots"


def run_forecaster_comparison(
    root: Optional[PathLike] = None,
    forecasters: Optional[Sequence[str]] = None,
    forecaster_root: Optional[PathLike] = None,
    train_dataset: Optional[str] = None,
    aggregate_only: bool = False,
    plot_periodic_only: bool = False,
    skip_periodic_plots: bool = False,
    periodic_repeats: int = 10,
    reporter_species: Optional[str] = None,
    visualization: Optional[bool] = None,
    include_main_periodic: Optional[str] = None,
    include_test_periodic: Optional[str] = None,
):
    """
    Run or aggregate one or more forecaster comparison recipes.

    Recipes are discovered under ``root/forecasters/*/recipe.json`` unless
    ``forecaster_root`` is provided. Dataset paths inside a recipe are resolved
    relative to ``root``.
    """
    experiment_root = Path(root or Path.cwd()).resolve()
    recipe_paths = selected_recipe_paths(
        root=experiment_root,
        forecasters=forecasters,
        forecaster_root=forecaster_root,
    )

    result = {
        "root": experiment_root,
        "recipes": recipe_paths,
        "periodic_plots": [],
        "runs": [],
        "aggregates": [],
    }

    if plot_periodic_only:
        recipe = load_recipe(recipe_paths[0])
        species = _recipe_reporter_species(recipe, reporter_species)
        dataset_paths = latest_dataset_paths(recipe["datasets"], experiment_root)
        result["periodic_plots"] = plot_periodic_sanity_panels(
            dataset_paths=dataset_paths,
            root=experiment_root,
            reporter_species=species,
            repeats=periodic_repeats,
        )
        return result

    if aggregate_only:
        for recipe_path in recipe_paths:
            result["aggregates"].append(aggregate_forecaster_recipe(recipe_path))
        return result

    if recipe_paths and not skip_periodic_plots and train_dataset is None:
        recipe = load_recipe(recipe_paths[0])
        species = _recipe_reporter_species(recipe, reporter_species)
        dataset_paths = latest_dataset_paths(recipe["datasets"], experiment_root)
        result["periodic_plots"] = plot_periodic_sanity_panels(
            dataset_paths=dataset_paths,
            root=experiment_root,
            reporter_species=species,
            repeats=periodic_repeats,
        )

    for recipe_path in recipe_paths:
        result["runs"].append(
            run_forecaster_recipe(
                recipe_path=recipe_path,
                root=experiment_root,
                train_dataset=train_dataset,
                reporter_species=reporter_species,
                visualization=visualization,
                include_main_periodic=include_main_periodic,
                include_test_periodic=include_test_periodic,
            )
        )

    return result


def selected_recipe_paths(root, forecasters=None, forecaster_root=None):
    root = Path(root or Path.cwd()).resolve()
    recipe_root = _resolve_path(forecaster_root or DEFAULT_FORECASTER_ROOT, base=root)
    recipe_paths = sorted(recipe_root.glob("*/recipe.json"))
    if not recipe_paths:
        raise FileNotFoundError(f"No forecaster recipe.json files found under {recipe_root}.")
    if not forecasters:
        return recipe_paths

    selected = []
    for forecaster in forecasters:
        direct = recipe_root / forecaster / "recipe.json"
        if direct.exists():
            selected.append(direct)
            continue

        matches = [
            path
            for path in recipe_paths
            if path.parent.name == forecaster or load_recipe(path).get("id") == forecaster
        ]
        if not matches:
            available = [path.parent.name for path in recipe_paths]
            raise FileNotFoundError(
                f"No forecaster recipe found for {forecaster!r}. Available: {available}"
            )
        selected.extend(matches)

    return sorted(dict.fromkeys(selected))


def run_forecaster_recipe(
    recipe_path,
    root=None,
    train_dataset=None,
    reporter_species=None,
    visualization=None,
    include_main_periodic=None,
    include_test_periodic=None,
):
    recipe_path = Path(recipe_path)
    recipe = load_recipe(recipe_path)
    validate_forecaster_recipe(recipe, recipe_path)

    experiment_root = Path(root).resolve() if root is not None else infer_experiment_root(recipe_path)
    forecaster_dir = recipe_path.parent
    datasets = recipe["datasets"]
    dataset_ids = [dataset_id(dataset) for dataset in datasets]
    dataset_paths = latest_dataset_paths(datasets, experiment_root)
    species = _recipe_reporter_species(recipe, reporter_species)
    model_config = reporter_only_model_config(recipe["forecaster_model"], dataset_paths, species)
    train_policy = _normalize_policy(
        include_main_periodic
        if include_main_periodic is not None
        else recipe.get("include_main_periodic", DEFAULT_INCLUDE_MAIN_PERIODIC)
    )
    test_policy = _normalize_policy(
        include_test_periodic
        if include_test_periodic is not None
        else recipe.get("include_test_periodic", DEFAULT_INCLUDE_TEST_PERIODIC)
    )
    use_visualization = (
        bool(visualization)
        if visualization is not None
        else bool(recipe.get("visualization", True))
    )

    print_model_setup(recipe["id"], model_config, train_policy, test_policy)

    if train_dataset is not None:
        if train_dataset not in dataset_ids:
            raise ValueError(
                f"Unknown train dataset {train_dataset!r}. Available datasets: {dataset_ids}"
            )
        return run_forecaster_row(
            recipe=recipe,
            forecaster_dir=forecaster_dir,
            dataset_ids=dataset_ids,
            dataset_paths=dataset_paths,
            model_config=model_config,
            train_id=train_dataset,
            include_main_periodic=train_policy,
            include_test_periodic=test_policy,
            visualization=use_visualization,
        )

    row_results = []
    for train_id in dataset_ids:
        row_results.append(
            run_forecaster_row(
                recipe=recipe,
                forecaster_dir=forecaster_dir,
                dataset_ids=dataset_ids,
                dataset_paths=dataset_paths,
                model_config=model_config,
                train_id=train_id,
                include_main_periodic=train_policy,
                include_test_periodic=test_policy,
                visualization=use_visualization,
            )
        )

    aggregate = aggregate_forecaster_recipe(recipe_path)
    print(f"Saved {recipe['id']} matrix to {forecaster_dir / 'mean_rmse_matrix.csv'}", flush=True)
    return {
        "recipe": recipe_path,
        "forecaster_id": recipe["id"],
        "rows": row_results,
        "aggregate": aggregate,
    }


def run_forecaster_row(
    recipe,
    forecaster_dir,
    dataset_ids,
    dataset_paths,
    model_config,
    train_id,
    include_main_periodic=DEFAULT_INCLUDE_MAIN_PERIODIC,
    include_test_periodic=DEFAULT_INCLUDE_TEST_PERIODIC,
    visualization=True,
):
    row_values = {test_id: None for test_id in dataset_ids}
    run_records = []
    trained_model = None

    for test_id in dataset_ids:
        label = f"{train_id}_to_{test_id}"
        output_root = Path(forecaster_dir) / "runs" / safe_filename(train_id)
        if trained_model is None:
            print(
                f"[{recipe['id']}] training on {train_id} "
                f"({dataset_paths[train_id]}) and testing on {test_id} "
                f"({dataset_paths[test_id]})",
                flush=True,
            )
            result = cross_test_forecaster(
                model=model_config,
                training_data=dataset_paths[train_id],
                test_data=dataset_paths[test_id],
                output_root=output_root,
                label=label,
                random_state=recipe.get("random_seed"),
                include_main_periodic=include_main_periodic,
                include_test_periodic=include_test_periodic,
                visualization=visualization,
            )
            trained_model = result["model_path"]
        else:
            print(
                f"[{recipe['id']}] reusing model trained on {train_id} "
                f"({trained_model}) and testing on {test_id} "
                f"({dataset_paths[test_id]})",
                flush=True,
            )
            result = cross_test_forecaster(
                trained_model=trained_model,
                test_data=dataset_paths[test_id],
                output_root=output_root,
                label=label,
                random_state=recipe.get("random_seed"),
                include_test_periodic=include_test_periodic,
                visualization=visualization,
            )

        test_metrics = result["metrics"]["test"]
        mean_rmse = test_metrics.get("rmse")
        finite_windows = test_metrics.get("finite_windows")
        invalid_windows = test_metrics.get("invalid_windows")
        print(
            f"[{recipe['id']}] completed train={train_id}, test={test_id}, "
            f"mean RMSE={_format_float(mean_rmse)}, "
            f"finite windows={finite_windows}, invalid windows={invalid_windows}",
            flush=True,
        )
        row_values[test_id] = mean_rmse
        run_records.append(
            {
                "train_dataset": train_id,
                "test_dataset": test_id,
                "mean_rmse": mean_rmse,
                "finite_windows": finite_windows,
                "invalid_windows": invalid_windows,
                "output_dir": result["output_dir"],
                "test_performance_path": result["test_performance_path"],
                "training_holdout_performance_path": result["training_holdout_performance_path"],
            }
        )

        save_row_outputs(forecaster_dir, train_id, dataset_ids, row_values, run_records)

    row_dir = Path(forecaster_dir) / "row_results"
    print(f"[{recipe['id']}] saved row results for train={train_id}", flush=True)
    return {
        "forecaster_id": recipe["id"],
        "train_dataset": train_id,
        "row_json": row_dir / f"{safe_filename(train_id)}.json",
        "row_csv": row_dir / f"{safe_filename(train_id)}.csv",
        "run_records": run_records,
    }


def aggregate_forecaster_recipe(recipe_path):
    recipe_path = Path(recipe_path)
    recipe = load_recipe(recipe_path)
    validate_forecaster_recipe(recipe, recipe_path)
    forecaster_dir = recipe_path.parent
    dataset_ids = [dataset_id(dataset) for dataset in recipe["datasets"]]
    row_dir = forecaster_dir / "row_results"
    matrix = pd.DataFrame(index=dataset_ids, columns=dataset_ids, dtype=float)
    run_records = []
    missing_rows = []

    for train_id in dataset_ids:
        row_path = row_dir / f"{safe_filename(train_id)}.json"
        if not row_path.exists():
            missing_rows.append(train_id)
            continue
        with open(row_path, "r") as f:
            row = json.load(f)
        values = row.get("mean_rmse_by_test", {})
        for test_id in dataset_ids:
            value = values.get(test_id)
            matrix.loc[train_id, test_id] = np.nan if value is None else float(value)
        run_records.extend(row.get("run_records", []))

    outputs = save_matrix_outputs(matrix, run_records, forecaster_dir, recipe["id"])
    if missing_rows:
        print(
            f"[{recipe['id']}] aggregated partial matrix; missing rows: {missing_rows}",
            flush=True,
        )
    else:
        print(f"[{recipe['id']}] aggregated complete matrix", flush=True)

    return {
        "recipe": recipe_path,
        "forecaster_id": recipe["id"],
        "missing_rows": missing_rows,
        "outputs": outputs,
    }


def load_recipe(recipe_path):
    with open(recipe_path, "r") as f:
        recipe = json.load(f)
    if not isinstance(recipe, dict):
        raise ValueError(f"{recipe_path} must contain a JSON object.")
    return recipe


def validate_forecaster_recipe(recipe, recipe_path=None):
    location = f" in {recipe_path}" if recipe_path is not None else ""
    if "id" not in recipe:
        raise ValueError(f"Forecaster recipe{location} must define `id`.")
    if "forecaster_model" not in recipe:
        raise ValueError(f"Forecaster recipe{location} must define `forecaster_model`.")
    if not isinstance(recipe.get("datasets"), list) or not recipe["datasets"]:
        raise ValueError(f"Forecaster recipe{location} must define a non-empty `datasets` list.")
    for dataset in recipe["datasets"]:
        dataset_id(dataset)
        dataset_path(dataset)


def latest_dataset_paths(datasets, root):
    root = Path(root or Path.cwd()).resolve()
    return {
        dataset_id(dataset): latest_simulation_file(_resolve_path(dataset_path(dataset), base=root))
        for dataset in datasets
    }


def latest_simulation_file(dataset_root):
    dataset_root = Path(dataset_root)
    if dataset_root.is_file():
        return dataset_root

    direct = dataset_root / "simulation.parquet"
    if direct.exists():
        return direct

    candidates = []
    for pattern in ("training_data/*/simulation.parquet", "*/simulation.parquet"):
        candidates.extend(dataset_root.glob(pattern))
    candidates = sorted(
        dict.fromkeys(candidates),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No simulation.parquet found under {dataset_root}. "
            "Run the simulation recipes first."
        )
    return candidates[0]


def reporter_only_model_config(model_config, dataset_paths, reporter_species=DEFAULT_REPORTER_SPECIES):
    model_config = copy.deepcopy(model_config)
    species = reporter_species or model_config.get("output_species") or DEFAULT_REPORTER_SPECIES
    model_config["feature_species"] = [species]
    model_config["output_species"] = species
    model_config.setdefault("past_input_window", model_config.get("past_feature_window"))
    model_config.setdefault("future_input_window", model_config.get("future_window"))
    species_by_dataset = {
        dataset_id: species_in_simulation_file(path)
        for dataset_id, path in dataset_paths.items()
    }
    validate_feature_species([species], species_by_dataset, species)
    return model_config


def print_model_setup(forecaster_id, model_config, include_main_periodic, include_test_periodic):
    model_type = model_config.get("type", model_config.get("model_type", "unknown"))
    print(
        f"[{forecaster_id}] model={model_type}; "
        f"past_feature_window={model_config.get('past_feature_window')}; "
        f"past_input_window={model_config.get('past_input_window')}; "
        f"future_window={model_config.get('future_window')}; "
        f"future_input_window={model_config.get('future_input_window')}; "
        f"feature_species={model_config['feature_species']}; "
        f"output_species={model_config['output_species']}; "
        f"input layout=past {model_config['output_species']} + past stim + future stim; "
        f"training periodic policy={include_main_periodic}; "
        f"test periodic policy={include_test_periodic}",
        flush=True,
    )


def validate_feature_species(feature_species, species_by_dataset, output_species):
    required_species = set(feature_species) | {output_species}
    missing_by_dataset = {
        dataset_id: sorted(required_species - set(species))
        for dataset_id, species in species_by_dataset.items()
        if required_species - set(species)
    }
    if missing_by_dataset:
        raise ValueError(
            "Configured feature/output species are not available in every dataset: "
            f"{missing_by_dataset}"
        )


def species_in_simulation_file(path):
    dataframe = pd.read_parquet(path, columns=["species"])
    return list(dict.fromkeys(dataframe["species"].astype(str)))


def save_row_outputs(forecaster_dir, train_id, dataset_ids, row_values, run_records):
    row_dir = Path(forecaster_dir) / "row_results"
    row_dir.mkdir(parents=True, exist_ok=True)
    row_label = safe_filename(train_id)

    row_frame = pd.DataFrame([row_values], index=[train_id], columns=dataset_ids, dtype=float)
    row_frame.to_csv(row_dir / f"{row_label}.csv")

    payload = {
        "train_dataset": train_id,
        "test_datasets": list(dataset_ids),
        "mean_rmse_by_test": row_values,
        "run_records": run_records,
    }
    with open(row_dir / f"{row_label}.json", "w") as f:
        json.dump(json_safe(payload), f, indent=2)


def save_matrix_outputs(matrix, run_records, forecaster_dir, forecaster_id):
    forecaster_dir = Path(forecaster_dir)
    outputs = {
        "csv": forecaster_dir / "mean_rmse_matrix.csv",
        "json": forecaster_dir / "mean_rmse_matrix.json",
        "runs": forecaster_dir / "cross_testing_runs.json",
        "svg": forecaster_dir / "mean_rmse_matrix.svg",
        "png": forecaster_dir / "mean_rmse_matrix.png",
    }

    matrix.to_csv(outputs["csv"])
    with open(outputs["json"], "w") as f:
        json.dump(json.loads(matrix.to_json(orient="index")), f, indent=2)
    with open(outputs["runs"], "w") as f:
        json.dump(json_safe(run_records), f, indent=2)

    title = f"{forecaster_id} cross-testing mean RMSE"
    plot_rmse_matrix(matrix, outputs["svg"], title)
    plot_rmse_matrix(matrix, outputs["png"], title)
    return outputs


def plot_rmse_matrix(matrix, output_path, title):
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(Path(output_path).parent.parent / ".matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    values = matrix.astype(float).to_numpy()
    masked_values = np.ma.masked_invalid(values)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#f1f1f1")

    fig_width = max(8, 0.9 * len(matrix.columns) + 3)
    fig_height = max(7, 0.75 * len(matrix.index) + 3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    image = ax.imshow(masked_values, cmap=cmap, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("Test dataset")
    ax.set_ylabel("Training dataset")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)

    finite_values = values[np.isfinite(values)]
    midpoint = float(np.nanmedian(finite_values)) if finite_values.size else 0.0
    for row_index, train_id in enumerate(matrix.index):
        for col_index, test_id in enumerate(matrix.columns):
            value = matrix.loc[train_id, test_id]
            if pd.isna(value):
                continue
            text_color = "white" if float(value) > midpoint else "black"
            ax.text(
                col_index,
                row_index,
                f"{float(value):.3g}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Mean RMSE")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_periodic_sanity_panels(
    dataset_paths,
    root=None,
    reporter_species=DEFAULT_REPORTER_SPECIES,
    repeats=10,
    output_root=None,
):
    output_root = Path(output_root or Path(root or Path.cwd()) / DEFAULT_PERIODIC_PLOTS_DIR)
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Saving periodic sanity plots to {output_root}", flush=True)

    outputs = []
    for group_name, title in (
        ("repetitive_stimulation_cells_red_first", "red-first periodic"),
        ("repetitive_stimulation_cells_green_first", "green-first periodic"),
    ):
        for suffix in ("svg", "png"):
            plot_path = output_root / f"{group_name}_first_{repeats}.{suffix}"
            plot_periodic_sanity_panel(
                dataset_paths=dataset_paths,
                group_name=group_name,
                title=title,
                repeats=repeats,
                output_path=plot_path,
                reporter_species=reporter_species,
            )
            outputs.append(plot_path)
    return outputs


def plot_periodic_sanity_panel(
    dataset_paths,
    group_name,
    title,
    repeats,
    output_path,
    reporter_species=DEFAULT_REPORTER_SPECIES,
):
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(Path(output_path).parent.parent / ".matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    dataset_items = list(dataset_paths.items())
    fig, axes = plt.subplots(
        nrows=len(dataset_items),
        ncols=repeats,
        figsize=(2.35 * repeats, 1.65 * len(dataset_items)),
        squeeze=False,
        sharex=True,
    )
    for row, (dataset_name, path) in enumerate(dataset_items):
        cell_ids = periodic_cell_ids(path, group_name, repeats)
        dataframe = pd.read_parquet(
            path,
            columns=["cell_id", "realization", "species", "stim", "time", "value"],
        )
        dataframe = dataframe[dataframe["species"] == reporter_species]
        for col in range(repeats):
            ax = axes[row][col]
            if col >= len(cell_ids):
                ax.axis("off")
                continue
            cell_id = cell_ids[col]
            cell_frame = dataframe[dataframe["cell_id"].astype(int) == int(cell_id)]
            plot_periodic_cell(ax, cell_frame)
            if row == 0:
                ax.set_title(f"repeat {col + 1}", fontsize=9)
            if col == 0:
                ax.set_ylabel(dataset_name, fontsize=8)
            if row == len(dataset_items) - 1:
                ax.set_xlabel("time (min)", fontsize=8)
            ax.tick_params(axis="both", labelsize=7)

    fig.suptitle(f"{title} {reporter_species} traces", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"Saved periodic sanity plot: {output_path}", flush=True)


def plot_periodic_cell(ax, cell_frame):
    if cell_frame.empty:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
        return
    grouped = list(cell_frame.groupby("realization", sort=True))
    first = grouped[0][1].sort_values("time")
    plot_stim_background(ax, first["time"].to_numpy(dtype=float), first["stim"].to_numpy(dtype=float))
    for _, realization_frame in grouped:
        realization_frame = realization_frame.sort_values("time")
        time = realization_frame["time"].to_numpy(dtype=float)
        values = realization_frame["value"].to_numpy(dtype=float)
        finite = np.isfinite(time) & np.isfinite(values)
        if finite.any():
            ax.plot(time[finite], values[finite], color="black", linewidth=0.8, alpha=0.75)


def plot_stim_background(ax, time, stim):
    if len(time) == 0:
        return
    step = np.median(np.diff(time)) if len(time) > 1 else 5
    half_step = float(step) / 2
    for t, value in zip(time, stim):
        color = "green" if value >= 0.5 else "red"
        ax.axvspan(t - half_step, t + half_step, color=color, alpha=0.12, linewidth=0)


def periodic_cell_ids(path, group_name, repeats):
    config_path = Path(path).parent / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        cell_range = config.get("simulated_cells", {}).get(group_name)
        if cell_range and len(cell_range) == 2:
            start, end = [int(value) for value in cell_range]
            return list(range(start, min(end, start + repeats - 1) + 1))

    max_cell_id = int(pd.read_parquet(path, columns=["cell_id"])["cell_id"].max())
    random_cells = max_cell_id - 100
    if group_name.endswith("red_first"):
        start = random_cells + 1
    else:
        start = random_cells + 51
    return list(range(start, start + repeats))


def dataset_id(dataset):
    if not isinstance(dataset, Mapping):
        raise ValueError(f"Dataset entries must be objects, got {dataset!r}.")
    value = dataset.get("id")
    if value is None:
        path = dataset_path(dataset)
        value = Path(path).name
    value = str(value)
    if not value:
        raise ValueError(f"Dataset id cannot be empty: {dataset!r}")
    return value


def dataset_path(dataset):
    for key in ("path", "simulation_path", "simulation_file", "data_path"):
        value = dataset.get(key)
        if value:
            return value
    raise ValueError(f"Dataset {dataset!r} must define a path.")


def infer_experiment_root(recipe_path):
    recipe_path = Path(recipe_path).resolve()
    if recipe_path.parent.parent.name == DEFAULT_FORECASTER_ROOT:
        return recipe_path.parent.parent.parent
    return recipe_path.parent


def safe_filename(value):
    return str(value).replace("/", "_").replace(" ", "_")


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, PathLike):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def add_cli_args(parser):
    parser.add_argument(
        "--root",
        default=None,
        help="Experiment root folder. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--forecaster-root",
        default=None,
        help="Folder containing forecaster recipe subfolders. Defaults to ROOT/forecasters.",
    )
    parser.add_argument(
        "--forecaster",
        action="append",
        help="Forecaster id/folder to run. Can be passed more than once. Defaults to all forecasters.",
    )
    parser.add_argument(
        "--train-dataset",
        help="Run only one training-dataset row. Intended for parallel row workers.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Aggregate row_results into matrix CSV/JSON/plots without running training.",
    )
    parser.add_argument(
        "--plot-periodic-only",
        action="store_true",
        help="Only create periodic sanity plots and exit.",
    )
    parser.add_argument(
        "--skip-periodic-plots",
        action="store_true",
        help="Skip periodic sanity plots during full matrix runs.",
    )
    parser.add_argument(
        "--periodic-repeats",
        type=int,
        default=10,
        help="Number of periodic cells to show per dataset in sanity plots.",
    )
    parser.add_argument(
        "--reporter-species",
        default=None,
        help="Reporter/output species used for features and predictions. Defaults to recipe value or F.",
    )
    parser.add_argument(
        "--include-main-periodic",
        choices=("train", "eval", "none"),
        default=None,
        help="Override the training-data periodic policy from the recipe.",
    )
    parser.add_argument(
        "--include-test-periodic",
        choices=("train", "eval", "none"),
        default=None,
        help="Override the test-data periodic policy from the recipe.",
    )
    parser.add_argument(
        "--visualization",
        dest="visualization",
        action="store_true",
        default=None,
        help="Force saving holdout/test prediction plots.",
    )
    parser.add_argument(
        "--no-visualization",
        dest="visualization",
        action="store_false",
        help="Force disabling holdout/test prediction plots.",
    )


def run_from_args(args):
    return run_forecaster_comparison(
        root=args.root,
        forecasters=args.forecaster,
        forecaster_root=args.forecaster_root,
        train_dataset=args.train_dataset,
        aggregate_only=args.aggregate_only,
        plot_periodic_only=args.plot_periodic_only,
        skip_periodic_plots=args.skip_periodic_plots,
        periodic_repeats=args.periodic_repeats,
        reporter_species=args.reporter_species,
        visualization=args.visualization,
        include_main_periodic=args.include_main_periodic,
        include_test_periodic=args.include_test_periodic,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and cross-test AISAM forecaster matrices.")
    add_cli_args(parser)
    args = parser.parse_args(argv)
    result = run_from_args(args)
    print(json.dumps(json_safe(result), indent=2))
    return 0


def _resolve_path(path, base):
    path = Path(path)
    return path if path.is_absolute() else Path(base) / path


def _recipe_reporter_species(recipe, reporter_species=None):
    if reporter_species:
        return reporter_species
    model_config = recipe.get("forecaster_model", {})
    return (
        recipe.get("reporter_species")
        or recipe.get("output_species")
        or model_config.get("output_species")
        or DEFAULT_REPORTER_SPECIES
    )


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


def _format_float(value):
    if value is None:
        return "nan"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{value:.6g}" if np.isfinite(value) else "nan"


if __name__ == "__main__":
    raise SystemExit(main())
