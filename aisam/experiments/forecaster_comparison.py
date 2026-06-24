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
    prepare_splits_only: bool = False,
    plot_forecaster_matrices_only: bool = False,
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
        "prepared_splits": [],
        "combined_matrix_plots": [],
    }

    if plot_forecaster_matrices_only:
        result["combined_matrix_plots"].append(
            plot_forecaster_rmse_matrices(recipe_paths=recipe_paths, root=experiment_root)
        )
        return result

    if prepare_splits_only:
        for recipe_path in recipe_paths:
            recipe = load_recipe(recipe_path)
            if _is_mixed_recipe(recipe):
                result["prepared_splits"].append(
                    prepare_mixed_splits(
                        recipe_path=recipe_path,
                        root=experiment_root,
                        reuse_existing=False,
                    )
                )
        return result

    if plot_periodic_only:
        recipe = load_recipe(recipe_paths[0])
        if _is_mixed_recipe(recipe):
            return result
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
        if len(recipe_paths) > 1:
            result["combined_matrix_plots"].append(
                plot_forecaster_rmse_matrices(recipe_paths=recipe_paths, root=experiment_root)
            )
        return result

    if recipe_paths and not skip_periodic_plots and train_dataset is None:
        recipe = load_recipe(recipe_paths[0])
        if not _is_mixed_recipe(recipe):
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

    if train_dataset is None and len(recipe_paths) > 1:
        result["combined_matrix_plots"].append(
            plot_forecaster_rmse_matrices(recipe_paths=recipe_paths, root=experiment_root)
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
    if _is_mixed_recipe(recipe):
        return run_mixed_forecaster_recipe(
            recipe_path=recipe_path,
            root=root,
            train_dataset=train_dataset,
            reporter_species=reporter_species,
            visualization=visualization,
            include_main_periodic=include_main_periodic,
            include_test_periodic=include_test_periodic,
        )

    experiment_root = Path(root).resolve() if root is not None else infer_experiment_root(recipe_path)
    forecaster_dir = recipe_path.parent
    datasets = recipe["datasets"]
    dataset_ids = [dataset_id(dataset) for dataset in datasets]
    dataset_paths = latest_dataset_paths(datasets, experiment_root)
    species = _recipe_reporter_species(recipe, reporter_species)
    model_config = reporter_only_model_config(
        recipe["forecaster_model"],
        dataset_paths,
        species,
        force_reporter_only=reporter_species is not None,
    )
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


def run_mixed_forecaster_recipe(
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
    split_info = prepare_mixed_splits(recipe_path=recipe_path, root=experiment_root, reuse_existing=True)
    groups = split_info["groups"]
    group_ids = [group["id"] for group in groups]
    test_ids = _mixed_test_ids(recipe)
    species = _recipe_reporter_species(recipe, reporter_species)
    model_config = reporter_only_model_config(
        recipe["forecaster_model"],
        _mixed_validation_paths(groups),
        species,
        force_reporter_only=reporter_species is not None,
    )
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
        selected = next((group for group in groups if group["id"] == train_dataset), None)
        if selected is None:
            raise ValueError(
                f"Unknown mixed train group {train_dataset!r}. Available groups: {group_ids}"
            )
        return run_mixed_forecaster_row(
            recipe=recipe,
            forecaster_dir=forecaster_dir,
            group=selected,
            test_ids=test_ids,
            model_config=model_config,
            include_main_periodic=train_policy,
            include_test_periodic=test_policy,
            visualization=use_visualization,
        )

    row_results = []
    for group in groups:
        row_results.append(
            run_mixed_forecaster_row(
                recipe=recipe,
                forecaster_dir=forecaster_dir,
                group=group,
                test_ids=test_ids,
                model_config=model_config,
                include_main_periodic=train_policy,
                include_test_periodic=test_policy,
                visualization=use_visualization,
            )
        )

    aggregate = aggregate_mixed_forecaster_recipe(recipe_path)
    print(f"Saved {recipe['id']} mixed matrix to {forecaster_dir / 'mean_rmse_matrix.csv'}", flush=True)
    return {
        "recipe": recipe_path,
        "forecaster_id": recipe["id"],
        "rows": row_results,
        "aggregate": aggregate,
        "split_info": split_info,
    }


def run_mixed_forecaster_row(
    recipe,
    forecaster_dir,
    group,
    test_ids,
    model_config,
    include_main_periodic=DEFAULT_INCLUDE_MAIN_PERIODIC,
    include_test_periodic=DEFAULT_INCLUDE_TEST_PERIODIC,
    visualization=True,
):
    group_id = group["id"]
    row_values = {test_id: None for test_id in test_ids}
    run_records = []
    trained_model = None
    training_paths = [Path(path) for path in group["training_paths"]]
    test_paths = {test_id: Path(path) for test_id, path in group["test_paths"].items()}

    for test_id in test_ids:
        label = f"{group_id}_to_{test_id}"
        output_root = Path(forecaster_dir) / "runs" / safe_filename(group_id)
        if trained_model is None:
            print(
                f"[{recipe['id']}] training on mixed group {group_id} "
                f"({[str(path) for path in training_paths]}) and testing on {test_id} "
                f"({test_paths[test_id]})",
                flush=True,
            )
            result = cross_test_forecaster(
                model=model_config,
                training_data=training_paths,
                test_data=test_paths[test_id],
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
                f"[{recipe['id']}] reusing model trained on mixed group {group_id} "
                f"({trained_model}) and testing on {test_id} ({test_paths[test_id]})",
                flush=True,
            )
            result = cross_test_forecaster(
                trained_model=trained_model,
                test_data=test_paths[test_id],
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
            f"[{recipe['id']}] completed train={group_id}, test={test_id}, "
            f"mean RMSE={_format_float(mean_rmse)}, "
            f"finite windows={finite_windows}, invalid windows={invalid_windows}",
            flush=True,
        )
        row_values[test_id] = mean_rmse
        run_records.append(
            {
                "train_dataset": group_id,
                "test_dataset": test_id,
                "training_paths": training_paths,
                "test_path": test_paths[test_id],
                "mean_rmse": mean_rmse,
                "finite_windows": finite_windows,
                "invalid_windows": invalid_windows,
                "output_dir": result["output_dir"],
                "test_performance_path": result["test_performance_path"],
                "training_holdout_performance_path": result["training_holdout_performance_path"],
            }
        )

        save_row_outputs(forecaster_dir, group_id, test_ids, row_values, run_records)

    row_dir = Path(forecaster_dir) / "row_results"
    print(f"[{recipe['id']}] saved mixed row results for train={group_id}", flush=True)
    return {
        "forecaster_id": recipe["id"],
        "train_dataset": group_id,
        "row_json": row_dir / f"{safe_filename(group_id)}.json",
        "row_csv": row_dir / f"{safe_filename(group_id)}.csv",
        "run_records": run_records,
    }


def aggregate_mixed_forecaster_recipe(recipe_path):
    recipe_path = Path(recipe_path)
    recipe = load_recipe(recipe_path)
    validate_forecaster_recipe(recipe, recipe_path)
    forecaster_dir = recipe_path.parent
    group_ids = [group["id"] for group in recipe["mixed_groups"]]
    test_ids = _mixed_test_ids(recipe)
    row_dir = forecaster_dir / "row_results"
    matrix = pd.DataFrame(index=group_ids, columns=test_ids, dtype=float)
    run_records = []
    missing_rows = []

    for group_id in group_ids:
        row_path = row_dir / f"{safe_filename(group_id)}.json"
        if not row_path.exists():
            missing_rows.append(group_id)
            continue
        with open(row_path, "r") as f:
            row = json.load(f)
        values = row.get("mean_rmse_by_test", {})
        for test_id in test_ids:
            value = values.get(test_id)
            matrix.loc[group_id, test_id] = np.nan if value is None else float(value)
        run_records.extend(row.get("run_records", []))

    outputs = save_matrix_outputs(matrix, run_records, forecaster_dir, recipe["id"])
    if missing_rows:
        print(
            f"[{recipe['id']}] aggregated partial mixed matrix; missing rows: {missing_rows}",
            flush=True,
        )
    else:
        print(f"[{recipe['id']}] aggregated complete mixed matrix", flush=True)

    return {
        "recipe": recipe_path,
        "forecaster_id": recipe["id"],
        "missing_rows": missing_rows,
        "outputs": outputs,
    }


def prepare_mixed_splits(recipe_path, root=None, reuse_existing=True):
    recipe_path = Path(recipe_path)
    recipe = load_recipe(recipe_path)
    validate_forecaster_recipe(recipe, recipe_path)
    if not _is_mixed_recipe(recipe):
        return {"recipe": recipe_path, "groups": []}

    experiment_root = Path(root).resolve() if root is not None else infer_experiment_root(recipe_path)
    split_config = recipe.get("split", {})
    split_root = _resolve_path(split_config.get("root", "splits"), base=experiment_root)
    test_fraction = float(split_config.get("test_fraction", 0.2))
    random_seed = int(split_config.get("random_seed", recipe.get("random_seed", 0)))
    if test_fraction <= 0 or test_fraction >= 1:
        raise ValueError("split.test_fraction must be in the range (0, 1).")

    if recipe.get("test_sources"):
        return _prepare_mixed_splits_with_test_sources(
            recipe=recipe,
            recipe_path=recipe_path,
            experiment_root=experiment_root,
            split_root=split_root,
            test_fraction=test_fraction,
            random_seed=random_seed,
            reuse_existing=reuse_existing,
        )

    prepared_groups = []
    for group_index, group in enumerate(recipe["mixed_groups"]):
        group_id = group["id"]
        training_paths = []
        test_paths = {}
        source_infos = []
        for source_index, source in enumerate(group["sources"]):
            source_id = dataset_id(source)
            test_id = source_test_id(source)
            source_path = latest_simulation_file(_resolve_path(dataset_path(source), base=experiment_root))
            source_split_root = split_root / safe_filename(group_id) / safe_filename(source_id)
            train_path = source_split_root / "train_pool" / "simulation.parquet"
            test_path = source_split_root / "individual_test" / "simulation.parquet"
            manifest_path = source_split_root / "split_manifest.json"
            if not reuse_existing or not (train_path.exists() and test_path.exists() and manifest_path.exists()):
                _write_random_cell_split(
                    source_path=source_path,
                    train_path=train_path,
                    test_path=test_path,
                    manifest_path=manifest_path,
                    source_id=source_id,
                    group_id=group_id,
                    test_fraction=test_fraction,
                    random_seed=random_seed + group_index * 1009 + source_index,
                )
            training_paths.append(train_path)
            test_paths[test_id] = test_path
            source_infos.append(
                {
                    "source_id": source_id,
                    "test_id": test_id,
                    "source_path": source_path,
                    "train_path": train_path,
                    "test_path": test_path,
                    "manifest_path": manifest_path,
                }
            )
        prepared_groups.append(
            {
                "id": group_id,
                "training_paths": training_paths,
                "test_paths": test_paths,
                "sources": source_infos,
            }
        )

    return {
        "recipe": recipe_path,
        "split_root": split_root,
        "groups": prepared_groups,
    }


def _prepare_mixed_splits_with_test_sources(
    recipe,
    recipe_path,
    experiment_root,
    split_root,
    test_fraction,
    random_seed,
    reuse_existing=True,
):
    source_defs = {}
    ordered_source_ids = []

    def add_source(source, source_seed):
        source_id = dataset_id(source)
        source_path = dataset_path(source)
        if source_id in source_defs:
            existing_path = dataset_path(source_defs[source_id]["source"])
            if str(existing_path) != str(source_path):
                raise ValueError(
                    f"Mixed source {source_id!r} is defined with multiple paths: "
                    f"{existing_path!r} and {source_path!r}."
                )
            return
        source_defs[source_id] = {
            "source": source,
            "random_seed": int(source_seed),
        }
        ordered_source_ids.append(source_id)

    for group_index, group in enumerate(recipe["mixed_groups"]):
        for source_index, source in enumerate(group["sources"]):
            add_source(source, random_seed + group_index * 1009 + source_index)

    extra_index = 0
    for source in recipe["test_sources"]:
        source_id = dataset_id(source)
        if source_id not in source_defs:
            add_source(source, random_seed + 50000 + extra_index)
        extra_index += 1

    source_splits = {}
    for source_id in ordered_source_ids:
        source_info = source_defs[source_id]
        source = source_info["source"]
        source_path = latest_simulation_file(_resolve_path(dataset_path(source), base=experiment_root))
        source_split_root = split_root / "sources" / safe_filename(source_id)
        train_path = source_split_root / "train_pool" / "simulation.parquet"
        test_path = source_split_root / "individual_test" / "simulation.parquet"
        manifest_path = source_split_root / "split_manifest.json"
        if not reuse_existing or not (train_path.exists() and test_path.exists() and manifest_path.exists()):
            _write_random_cell_split(
                source_path=source_path,
                train_path=train_path,
                test_path=test_path,
                manifest_path=manifest_path,
                source_id=source_id,
                group_id="shared_mixed_sources",
                test_fraction=test_fraction,
                random_seed=source_info["random_seed"],
            )
        source_splits[source_id] = {
            "source_id": source_id,
            "source_path": source_path,
            "train_path": train_path,
            "test_path": test_path,
            "manifest_path": manifest_path,
            "random_seed": source_info["random_seed"],
        }

    test_source_infos = []
    test_paths = {}
    for source in recipe["test_sources"]:
        source_id = dataset_id(source)
        test_id = source_test_id(source)
        split_info = source_splits[source_id]
        test_paths[test_id] = split_info["test_path"]
        test_source_infos.append(
            {
                "source_id": source_id,
                "test_id": test_id,
                "source_path": split_info["source_path"],
                "train_path": split_info["train_path"],
                "test_path": split_info["test_path"],
                "manifest_path": split_info["manifest_path"],
                "random_seed": split_info["random_seed"],
            }
        )

    prepared_groups = []
    for group in recipe["mixed_groups"]:
        group_id = group["id"]
        training_paths = []
        source_infos = []
        for source in group["sources"]:
            source_id = dataset_id(source)
            split_info = source_splits[source_id]
            training_paths.append(split_info["train_path"])
            source_infos.append(
                {
                    "source_id": source_id,
                    "source_path": split_info["source_path"],
                    "train_path": split_info["train_path"],
                    "test_path": split_info["test_path"],
                    "manifest_path": split_info["manifest_path"],
                    "random_seed": split_info["random_seed"],
                }
            )
        prepared_groups.append(
            {
                "id": group_id,
                "training_paths": training_paths,
                "test_paths": dict(test_paths),
                "sources": source_infos,
                "test_sources": test_source_infos,
            }
        )

    return {
        "recipe": recipe_path,
        "split_root": split_root,
        "groups": prepared_groups,
        "test_sources": test_source_infos,
        "source_splits": source_splits,
    }


def aggregate_forecaster_recipe(recipe_path):
    recipe_path = Path(recipe_path)
    recipe = load_recipe(recipe_path)
    validate_forecaster_recipe(recipe, recipe_path)
    if _is_mixed_recipe(recipe):
        return aggregate_mixed_forecaster_recipe(recipe_path)

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
    if _is_mixed_recipe(recipe):
        if not isinstance(recipe.get("mixed_groups"), list) or not recipe["mixed_groups"]:
            raise ValueError(f"Mixed forecaster recipe{location} must define a non-empty `mixed_groups` list.")
        if "test_sources" in recipe:
            if not isinstance(recipe.get("test_sources"), list) or not recipe["test_sources"]:
                raise ValueError(f"Mixed forecaster recipe{location} must define a non-empty `test_sources` list.")
            test_ids = []
            for source in recipe["test_sources"]:
                dataset_id(source)
                dataset_path(source)
                test_id = source_test_id(source)
                if test_id in test_ids:
                    raise ValueError(
                        f"Mixed forecaster recipe{location} has duplicate test source label {test_id!r}."
                    )
                test_ids.append(test_id)
        for group in recipe["mixed_groups"]:
            if not isinstance(group, Mapping):
                raise ValueError(f"Mixed group entries{location} must be objects.")
            if not group.get("id"):
                raise ValueError(f"Mixed group entries{location} must define `id`.")
            if not isinstance(group.get("sources"), list) or len(group["sources"]) < 2:
                raise ValueError(f"Mixed group {group.get('id')!r}{location} must define at least two sources.")
            for source in group["sources"]:
                dataset_id(source)
                dataset_path(source)
        return
    if not isinstance(recipe.get("datasets"), list) or not recipe["datasets"]:
        raise ValueError(f"Forecaster recipe{location} must define a non-empty `datasets` list.")
    for dataset in recipe["datasets"]:
        dataset_id(dataset)
        dataset_path(dataset)


def _is_mixed_recipe(recipe):
    mode = str(recipe.get("mode", "")).strip().lower().replace("-", "_").replace(" ", "_")
    return mode in {"mixed_cross_testing", "mixed_cross_testing_matrix"} or "mixed_groups" in recipe


def _mixed_test_ids(recipe):
    if recipe.get("test_sources"):
        return [source_test_id(source) for source in recipe["test_sources"]]

    test_ids = []
    for group in recipe.get("mixed_groups", []):
        for source in group.get("sources", []):
            test_id = source_test_id(source)
            if test_id not in test_ids:
                test_ids.append(test_id)
    return test_ids


def _mixed_validation_paths(groups):
    paths = {}
    for group in groups:
        for index, path in enumerate(group.get("training_paths", []), start=1):
            paths[f"{group['id']}_train_{index}"] = path
        for test_id, path in group.get("test_paths", {}).items():
            paths[f"{group['id']}_test_{test_id}"] = path
    return paths


def _write_random_cell_split(
    source_path,
    train_path,
    test_path,
    manifest_path,
    source_id,
    group_id,
    test_fraction,
    random_seed,
):
    source_path = Path(source_path)
    dataframe = pd.read_parquet(source_path)
    random_cell_ids = _source_random_cell_ids(source_path, dataframe)
    rng = np.random.default_rng(random_seed)
    shuffled = np.array(random_cell_ids, dtype=object)
    rng.shuffle(shuffled)
    num_test = int(np.ceil(len(shuffled) * float(test_fraction)))
    num_test = min(max(num_test, 1), len(shuffled) - 1)
    test_draw_order = shuffled[:num_test].tolist()
    train_draw_order = shuffled[num_test:].tolist()
    test_ids = set(test_draw_order)
    train_ids = set(train_draw_order)

    train_frame = dataframe[dataframe["cell_id"].isin(train_ids)].copy()
    test_frame = dataframe[dataframe["cell_id"].isin(test_ids)].copy()
    train_mapping = _remap_cell_ids(train_frame)
    test_mapping = _remap_cell_ids(test_frame)
    _write_split_dataset(train_frame, train_path, source_path, len(train_mapping))
    _write_split_dataset(test_frame, test_path, source_path, len(test_mapping))

    manifest = {
        "group_id": group_id,
        "source_id": source_id,
        "source_path": source_path,
        "split_method": "seeded_random_shuffle_by_cell_id",
        "random_stimulation_only": True,
        "test_fraction": test_fraction,
        "random_seed": random_seed,
        "random_source_cells": len(random_cell_ids),
        "train_cells": len(train_mapping),
        "test_cells": len(test_mapping),
        "train_path": train_path,
        "test_path": test_path,
        "train_source_cell_ids_draw_order": train_draw_order,
        "test_source_cell_ids_draw_order": test_draw_order,
        "train_cell_id_map": train_mapping,
        "test_cell_id_map": test_mapping,
    }
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(json_safe(manifest), f, indent=2)


def _source_random_cell_ids(source_path, dataframe):
    config = _source_run_config(source_path)
    simulated_cells = config.get("simulated_cells", {})
    random_range = simulated_cells.get("random_stimulation_cells")
    cell_ids = list(dict.fromkeys(dataframe["cell_id"].tolist()))
    if random_range and len(random_range) == 2:
        start, end = [int(value) for value in random_range]
        return [
            cell_id
            for cell_id in cell_ids
            if start <= int(cell_id) <= end
        ]

    numeric_ids = sorted(int(cell_id) for cell_id in cell_ids)
    total_cells = max(numeric_ids)
    random_end = total_cells - 100 if total_cells >= 200 else total_cells
    return [cell_id for cell_id in cell_ids if int(cell_id) <= random_end]


def _source_run_config(source_path):
    config_path = Path(source_path).parent / "config.json"
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return json.load(f)


def _remap_cell_ids(dataframe):
    original_ids = sorted(
        dict.fromkeys(dataframe["cell_id"].tolist()),
        key=lambda value: int(value),
    )
    mapping = {cell_id: index + 1 for index, cell_id in enumerate(original_ids)}
    dataframe["source_cell_id"] = dataframe["cell_id"]
    dataframe["cell_id"] = dataframe["cell_id"].map(mapping).astype(int)
    return {str(source): target for source, target in mapping.items()}


def _write_split_dataset(dataframe, output_path, source_path, total_cells):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(output_path, index=False)
    _write_split_config(output_path, source_path, total_cells)


def _write_split_config(output_path, source_path, total_cells):
    output_path = Path(output_path)
    source_config = _source_run_config(source_path)
    source_simulated = source_config.get("simulated_cells", {})
    config = {
        "source_simulation_file": str(source_path),
        "simulation_file": str(output_path),
        "simulation_data_file": str(output_path),
        "simulation_parquet_file": str(output_path),
        "data_format": "parquet_long_v1",
        "data_columns": ["cell_id", "realization", "species", "stim", "time", "value", "source_cell_id"],
        "simulated_cells": {
            "total_cells": int(total_cells),
            "random_stimulation_cells": [1, int(total_cells)],
            "repetitive_stimulation_cells_red_first": [],
            "repetitive_stimulation_cells_green_first": [],
            "include_repetitive_stims": False,
            "num_realizations": source_simulated.get("num_realizations"),
            "t_max": source_simulated.get("t_max"),
            "sampling": source_simulated.get("sampling"),
            "saved_sampling": source_simulated.get("saved_sampling"),
            "sample_interval_minutes": source_simulated.get("sample_interval_minutes"),
        },
        "source_config": source_config,
    }
    with open(output_path.parent / "config.json", "w") as f:
        json.dump(json_safe(config), f, indent=2)


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


def reporter_only_model_config(
    model_config,
    dataset_paths,
    reporter_species=DEFAULT_REPORTER_SPECIES,
    force_reporter_only=False,
):
    model_config = copy.deepcopy(model_config)
    output_species = reporter_species or model_config.get("output_species") or DEFAULT_REPORTER_SPECIES
    if force_reporter_only:
        feature_species = [output_species]
    else:
        feature_species = model_config.get("feature_species") or [output_species]
        if isinstance(feature_species, str):
            feature_species = [feature_species]
        else:
            feature_species = list(feature_species)
    model_config["feature_species"] = feature_species
    model_config["output_species"] = output_species
    model_config.setdefault("past_input_window", model_config.get("past_feature_window"))
    model_config.setdefault("future_input_window", model_config.get("future_window"))
    species_by_dataset = {
        dataset_id: species_in_simulation_file(path)
        for dataset_id, path in dataset_paths.items()
    }
    validate_feature_species(feature_species, species_by_dataset, output_species)
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
        f"input layout=past feature_species + past stim + future stim; "
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


def plot_forecaster_rmse_matrices(recipe_paths, root=None, output_prefix="forecaster_rmse_matrices"):
    """
    Plot all selected forecaster RMSE matrices side by side with one global color scale.

    Each recipe must already have a ``mean_rmse_matrix.csv`` in its forecaster
    directory. This makes the function usable after full runs, after
    aggregation, or as a plotting-only pass over existing experiment folders.
    """
    recipe_paths = [Path(path) for path in recipe_paths]
    if not recipe_paths:
        raise ValueError("At least one forecaster recipe is required for combined matrix plotting.")

    entries = []
    missing = []
    for recipe_path in recipe_paths:
        recipe = load_recipe(recipe_path)
        forecaster_id = str(recipe.get("id", recipe_path.parent.name))
        matrix_path = recipe_path.parent / "mean_rmse_matrix.csv"
        if not matrix_path.exists():
            missing.append(matrix_path)
            continue
        matrix = pd.read_csv(matrix_path, index_col=0)
        matrix = matrix.apply(pd.to_numeric, errors="coerce")
        entries.append(
            {
                "forecaster_id": forecaster_id,
                "matrix": matrix,
                "matrix_path": matrix_path,
            }
        )

    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Cannot plot combined forecaster matrix panel because these files are missing: "
            f"{missing_text}. Run aggregation first."
        )
    if not entries:
        raise ValueError("No forecaster RMSE matrices were available to plot.")

    output_root = Path(root).resolve() if root is not None else infer_experiment_root(recipe_paths[0])
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "svg": output_root / f"{output_prefix}.svg",
        "png": output_root / f"{output_prefix}.png",
    }
    title = "Forecaster cross-testing mean RMSE"
    scale = _global_rmse_scale([entry["matrix"] for entry in entries])
    for output_path in outputs.values():
        plot_forecaster_rmse_matrix_panel(
            entries=entries,
            output_path=output_path,
            title=title,
            vmin=scale["vmin"],
            vmax=scale["vmax"],
            midpoint=scale["midpoint"],
        )

    return {
        "outputs": outputs,
        "forecasters": [entry["forecaster_id"] for entry in entries],
        "matrix_paths": [entry["matrix_path"] for entry in entries],
        "vmin": scale["vmin"],
        "vmax": scale["vmax"],
        "midpoint": scale["midpoint"],
    }


def _global_rmse_scale(matrices):
    finite_chunks = []
    for matrix in matrices:
        values = matrix.astype(float).to_numpy()
        finite = values[np.isfinite(values)]
        if finite.size:
            finite_chunks.append(finite)

    if not finite_chunks:
        return {"vmin": 0.0, "vmax": 1.0, "midpoint": 0.5}

    finite_values = np.concatenate(finite_chunks)
    vmin = float(np.nanmin(finite_values))
    vmax = float(np.nanmax(finite_values))
    if vmin == vmax:
        pad = max(1.0, abs(vmin) * 0.05)
        vmin -= pad
        vmax += pad
    midpoint = float(np.nanmedian(finite_values))
    return {"vmin": vmin, "vmax": vmax, "midpoint": midpoint}


def plot_forecaster_rmse_matrix_panel(entries, output_path, title, vmin=None, vmax=None, midpoint=None):
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(Path(output_path).parent / ".matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    output_path = Path(output_path)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#f1f1f1")

    max_rows = max(len(entry["matrix"].index) for entry in entries)
    max_cols = max(len(entry["matrix"].columns) for entry in entries)
    panel_width = max(4.2, 0.58 * max_cols + 1.8)
    fig_width = max(8.0, panel_width * len(entries) + 1.2)
    fig_height = max(4.8, 0.62 * max_rows + 2.7)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(entries),
        figsize=(fig_width, fig_height),
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes[0]

    image = None
    for axis_index, (ax, entry) in enumerate(zip(axes, entries)):
        matrix = entry["matrix"].astype(float)
        values = matrix.to_numpy()
        image = ax.imshow(
            np.ma.masked_invalid(values),
            cmap=cmap,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(entry["forecaster_id"])
        ax.set_xlabel("Test dataset")
        ax.set_xticks(np.arange(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(matrix.index)))
        ax.set_yticklabels(matrix.index)
        if axis_index == 0:
            ax.set_ylabel("Training dataset")

        annotate_values = values.size <= 144
        if annotate_values:
            threshold = midpoint
            if threshold is None:
                finite_values = values[np.isfinite(values)]
                threshold = float(np.nanmedian(finite_values)) if finite_values.size else 0.0
            for row_index, train_id in enumerate(matrix.index):
                for col_index, test_id in enumerate(matrix.columns):
                    value = matrix.loc[train_id, test_id]
                    if pd.isna(value):
                        continue
                    text_color = "white" if float(value) > threshold else "black"
                    ax.text(
                        col_index,
                        row_index,
                        f"{float(value):.3g}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=7,
                    )

    fig.suptitle(title)
    colorbar = fig.colorbar(image, ax=list(axes), shrink=0.88)
    colorbar.set_label("Mean RMSE")
    fig.savefig(output_path, dpi=220)
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


def source_test_id(source):
    return str(source.get("test_id", source.get("test_label", dataset_id(source))))


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
        "--prepare-splits-only",
        action="store_true",
        help="Prepare mixed-comparison split parquet files and exit.",
    )
    parser.add_argument(
        "--plot-forecaster-matrices-only",
        action="store_true",
        help=(
            "Plot selected forecaster mean_rmse_matrix.csv files side by side "
            "with one global heatmap scale and exit."
        ),
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
        prepare_splits_only=args.prepare_splits_only,
        plot_forecaster_matrices_only=args.plot_forecaster_matrices_only,
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
