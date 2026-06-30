"""
Forecaster training-data size and balance experiments.

This module supports three recipe-driven experiment families:

1. Sample-efficiency sweeps: train on nested random-cell subsets from one
   dataset and test on a fixed held-out random-cell set.
2. Mixed-balance sweeps: train on two-circuit mixtures with a fixed total cell
   count and test on the fixed held-out data for each source circuit.
3. Simplex sweeps: train on three-circuit mixtures across a composition lattice
   and test on the fixed held-out data for all three source circuits.
"""

from __future__ import annotations

import argparse
import copy
import json
from os import PathLike
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from aisam.experiments.forecaster_comparison import (
    _remap_cell_ids,
    _resolve_path,
    _source_random_cell_ids,
    _write_split_dataset,
    dataset_id,
    dataset_path,
    json_safe,
    latest_simulation_file,
    load_recipe,
    reporter_only_model_config,
    safe_filename,
)
from aisam.utils.forecaster_training import cross_test_forecaster


DEFAULT_RECIPE = "recipe.json"
DEFAULT_REPORTER_SPECIES = "F"
DEFAULT_INCLUDE_MAIN_PERIODIC = "none"
DEFAULT_INCLUDE_TEST_PERIODIC = "none"
DEFAULT_SPLIT_ROOT = "splits"
DEFAULT_SAMPLE_ROOT = "sample_efficiency"
DEFAULT_BALANCE_ROOT = "balance"
DEFAULT_SIMPLEX_ROOT = "simplex"
DEFAULT_CIRCUITS = ("CcaSR", "Inverter", "double_inverter")

SAMPLE_SUMMARY_COLUMNS = (
    "mode",
    "forecaster_id",
    "dataset_id",
    "circuit",
    "solver",
    "fraction",
    "replicate",
    "mean_rmse",
    "train_cells",
    "test_cells",
)
SAMPLE_AGGREGATE_COLUMNS = (
    "forecaster_id",
    "dataset_id",
    "circuit",
    "solver",
    "fraction",
    "mean_rmse_mean",
    "mean_rmse_std",
    "mean_rmse_count",
    "train_cells_mean",
    "test_cells_mean",
)
BALANCE_SUMMARY_COLUMNS = (
    "mode",
    "forecaster_id",
    "solver",
    "pair_id",
    "left_circuit",
    "right_circuit",
    "test_dataset_id",
    "test_circuit",
    "ratio_left",
    "ratio_right",
    "replicate",
    "mean_rmse",
    "total_cells",
    "left_cells",
    "right_cells",
)
BALANCE_AGGREGATE_COLUMNS = (
    "forecaster_id",
    "solver",
    "pair_id",
    "left_circuit",
    "right_circuit",
    "test_dataset_id",
    "test_circuit",
    "ratio_left",
    "ratio_right",
    "mean_rmse_mean",
    "mean_rmse_std",
    "mean_rmse_count",
    "total_cells_mean",
    "left_cells_mean",
    "right_cells_mean",
)
SIMPLEX_SUMMARY_COLUMNS = (
    "mode",
    "forecaster_id",
    "solver",
    "composition_id",
    "test_dataset_id",
    "test_circuit",
    "fraction_1",
    "fraction_2",
    "fraction_3",
    "replicate",
    "mean_rmse",
    "total_cells",
    "cells_1",
    "cells_2",
    "cells_3",
)
SIMPLEX_AGGREGATE_COLUMNS = (
    "forecaster_id",
    "solver",
    "composition_id",
    "test_dataset_id",
    "test_circuit",
    "fraction_1",
    "fraction_2",
    "fraction_3",
    "mean_rmse_mean",
    "mean_rmse_std",
    "mean_rmse_count",
    "total_cells_mean",
    "cells_1_mean",
    "cells_2_mean",
    "cells_3_mean",
)


def run_training_data_experiment(
    root: Optional[PathLike] = None,
    recipe_path: Optional[PathLike] = None,
    validate_only: bool = False,
    prepare_sample_splits_only: bool = False,
    prepare_balance_splits_only: bool = False,
    prepare_simplex_splits_only: bool = False,
    run_sample: bool = False,
    run_balance: bool = False,
    run_simplex: bool = False,
    aggregate_only: bool = False,
    aggregate_simplex_only: bool = False,
    plot_only: bool = False,
    forecaster: Optional[str] = None,
    dataset: Optional[str] = None,
    fraction: Optional[float] = None,
    replicate: Optional[int] = None,
    solver: Optional[str] = None,
    pair: Optional[str] = None,
    ratio: Optional[float] = None,
    composition=None,
    balance_total_fraction: Optional[float] = None,
    visualization: Optional[bool] = None,
):
    experiment_root = Path(root or Path.cwd()).resolve()
    recipe_file = Path(recipe_path) if recipe_path is not None else experiment_root / DEFAULT_RECIPE
    recipe_file = recipe_file if recipe_file.is_absolute() else experiment_root / recipe_file
    recipe = load_recipe(recipe_file)
    validate_training_data_recipe(recipe, recipe_file)

    result = {
        "root": experiment_root,
        "recipe": recipe_file,
        "validated": True,
        "prepared_sample_splits": None,
        "prepared_balance_splits": None,
        "prepared_simplex_splits": None,
        "sample_run": None,
        "balance_run": None,
        "simplex_run": None,
        "aggregates": None,
        "plots": None,
    }

    if validate_only:
        validate_source_paths(recipe, experiment_root)
        return result

    if prepare_sample_splits_only:
        prepared = prepare_sample_splits(recipe, experiment_root, reuse_existing=False)
        result["prepared_sample_splits"] = summarize_sample_splits(prepared)
        return result

    if prepare_balance_splits_only:
        prepare_sample_splits(recipe, experiment_root, reuse_existing=True)
        prepared = prepare_balance_splits(
            recipe,
            experiment_root,
            reuse_existing=False,
            override_total_fraction=balance_total_fraction,
        )
        result["prepared_balance_splits"] = summarize_balance_splits(prepared)
        return result

    if prepare_simplex_splits_only:
        prepare_sample_splits(recipe, experiment_root, reuse_existing=True)
        prepared = prepare_simplex_splits(recipe, experiment_root, reuse_existing=True)
        result["prepared_simplex_splits"] = summarize_simplex_splits(prepared)
        return result

    if run_sample:
        require_values(
            forecaster=forecaster,
            dataset=dataset,
            fraction=fraction,
            replicate=replicate,
        )
        prepare_sample_splits(recipe, experiment_root, reuse_existing=True)
        result["sample_run"] = run_sample_efficiency_cell(
            recipe,
            experiment_root,
            forecaster_id=forecaster,
            dataset_id_value=dataset,
            fraction=float(fraction),
            replicate=int(replicate),
            visualization=visualization,
        )
        return result

    if run_balance:
        require_values(
            forecaster=forecaster,
            solver=solver,
            pair=pair,
            ratio=ratio,
            replicate=replicate,
        )
        prepare_sample_splits(recipe, experiment_root, reuse_existing=True)
        prepare_balance_splits(
            recipe,
            experiment_root,
            reuse_existing=True,
            override_total_fraction=balance_total_fraction,
        )
        result["balance_run"] = run_balance_cell(
            recipe,
            experiment_root,
            forecaster_id=forecaster,
            solver=solver,
            pair_id=pair,
            ratio=float(ratio),
            replicate=int(replicate),
            visualization=visualization,
        )
        return result

    if run_simplex:
        require_values(
            forecaster=forecaster,
            solver=solver,
            composition=composition,
            replicate=replicate,
        )
        result["simplex_run"] = run_simplex_cell(
            recipe,
            experiment_root,
            forecaster_id=forecaster,
            solver=solver,
            composition=parse_composition(composition),
            replicate=int(replicate),
            visualization=visualization,
        )
        return result

    if aggregate_only:
        result["aggregates"] = aggregate_training_data_results(recipe, experiment_root)
        result["plots"] = plot_training_data_results(recipe, experiment_root)
        return result

    if aggregate_simplex_only:
        result["aggregates"] = {"simplex": aggregate_simplex_results(recipe, experiment_root)}
        result["plots"] = {"simplex": plot_simplex_results(recipe, experiment_root)}
        return result

    if plot_only:
        result["plots"] = plot_training_data_results(recipe, experiment_root)
        return result

    return result


def validate_training_data_recipe(recipe, recipe_path=None):
    location = f" in {recipe_path}" if recipe_path is not None else ""
    if "id" not in recipe:
        raise ValueError(f"Training-data recipe{location} must define `id`.")
    if not isinstance(recipe.get("datasets"), list) or not recipe["datasets"]:
        raise ValueError(f"Training-data recipe{location} must define non-empty `datasets`.")
    if not isinstance(recipe.get("forecasters"), list) or not recipe["forecasters"]:
        raise ValueError(f"Training-data recipe{location} must define non-empty `forecasters`.")

    dataset_ids = []
    circuit_solver_pairs = set()
    for dataset in recipe["datasets"]:
        dataset_ids.append(dataset_id(dataset))
        dataset_path(dataset)
        if not dataset.get("circuit") or not dataset.get("solver"):
            raise ValueError(f"Dataset entries{location} must define `circuit` and `solver`.")
        key = (str(dataset["circuit"]), str(dataset["solver"]))
        if key in circuit_solver_pairs:
            raise ValueError(f"Duplicate circuit/solver dataset{location}: {key}.")
        circuit_solver_pairs.add(key)
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError(f"Dataset ids{location} must be unique.")

    for forecaster in recipe["forecasters"]:
        if not forecaster.get("id"):
            raise ValueError(f"Forecaster entries{location} must define `id`.")
        if "model" not in forecaster:
            raise ValueError(f"Forecaster {forecaster.get('id')!r}{location} must define `model`.")

    fractions = recipe.get("sample_efficiency", {}).get("fractions")
    if not fractions:
        raise ValueError(f"Recipe{location} must define sample_efficiency.fractions.")
    for value in fractions:
        fraction = float(value)
        if fraction <= 0 or fraction >= 1:
            raise ValueError("Sample-efficiency fractions must be in the range (0, 1).")

    ratios = recipe.get("balance", {}).get("ratios")
    if not ratios:
        raise ValueError(f"Recipe{location} must define balance.ratios.")
    for value in ratios:
        ratio = float(value)
        if ratio <= 0 or ratio >= 1:
            raise ValueError("Balance ratios must be in the range (0, 1).")

    simplex = recipe.get("simplex", {})
    circuits = simplex.get("circuits", recipe.get("balance", {}).get("circuits", DEFAULT_CIRCUITS))
    if len(circuits) != 3 or len(set(circuits)) != 3:
        raise ValueError(f"Recipe{location} simplex.circuits must contain exactly three unique circuits.")
    step = float(simplex.get("step", 0.1))
    minimum = float(simplex.get("minimum_fraction", step))
    simplex_compositions(step=step, minimum_fraction=minimum)


def validate_source_paths(recipe, experiment_root):
    paths = {}
    for dataset in recipe["datasets"]:
        source_path = latest_simulation_file(_resolve_path(dataset_path(dataset), base=experiment_root))
        paths[dataset_id(dataset)] = source_path
    return paths


def require_values(**values):
    missing = [key for key, value in values.items() if value is None]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")


def prepare_sample_splits(recipe, experiment_root, reuse_existing=True):
    split_root = split_root_path(recipe, experiment_root)
    source_infos = {}
    for dataset_index, dataset in enumerate(recipe["datasets"]):
        source_info = prepare_sample_source_splits(
            recipe=recipe,
            experiment_root=experiment_root,
            split_root=split_root,
            dataset=dataset,
            dataset_index=dataset_index,
            reuse_existing=reuse_existing,
        )
        source_infos[source_info["dataset_id"]] = source_info
    return {
        "split_root": split_root,
        "sources": source_infos,
    }


def summarize_sample_splits(prepared):
    sources = prepared.get("sources", {})
    train_split_count = sum(len(source.get("train_manifests", [])) for source in sources.values())
    return {
        "split_root": prepared.get("split_root"),
        "source_count": len(sources),
        "train_split_count": train_split_count,
        "dataset_ids": sorted(sources),
    }


def prepare_sample_source_splits(
    recipe,
    experiment_root,
    split_root,
    dataset,
    dataset_index,
    reuse_existing=True,
):
    source_id = dataset_id(dataset)
    source_path = latest_simulation_file(_resolve_path(dataset_path(dataset), base=experiment_root))
    source_root = split_root / "sources" / safe_filename(source_id)
    source_manifest_path = source_root / "source_manifest.json"
    fractions = sample_fractions(recipe)
    replicates = replicate_values(recipe)
    expected_paths = [source_root / "fixed_test" / "simulation.parquet", source_manifest_path]
    for replicate in replicates:
        for fraction in fractions:
            expected_paths.append(sample_train_path(split_root, source_id, fraction, replicate))

    if reuse_existing and all(Path(path).exists() for path in expected_paths):
        return load_json(source_manifest_path)

    dataframe = pd.read_parquet(source_path)
    random_cell_ids = list(_source_random_cell_ids(source_path, dataframe))
    if len(random_cell_ids) < 3:
        raise ValueError(f"Dataset {source_id!r} has too few random-stimulation cells.")

    seed = int(recipe.get("random_seed", 0)) + dataset_index * 1009
    rng = np.random.default_rng(seed)
    shuffled = np.array(random_cell_ids, dtype=object)
    rng.shuffle(shuffled)

    test_fraction = fixed_test_fraction(recipe)
    num_test = int(np.ceil(len(shuffled) * test_fraction))
    num_test = min(max(num_test, 1), len(shuffled) - 1)
    test_ids = shuffled[:num_test].tolist()
    train_pool_ids = shuffled[num_test:].tolist()

    test_path = source_root / "fixed_test" / "simulation.parquet"
    test_mapping = write_cell_subset(dataframe, test_ids, test_path, source_path)

    train_manifests = []
    for replicate in replicates:
        rep_seed = seed + int(replicate) * 7919
        rep_rng = np.random.default_rng(rep_seed)
        rep_pool = np.array(train_pool_ids, dtype=object)
        rep_rng.shuffle(rep_pool)
        rep_pool_order = rep_pool.tolist()
        for fraction in fractions:
            requested_cells = int(np.ceil(len(random_cell_ids) * float(fraction)))
            selected_count = min(max(requested_cells, 1), len(rep_pool_order))
            selected_ids = rep_pool_order[:selected_count]
            train_path = sample_train_path(split_root, source_id, fraction, replicate)
            mapping = write_cell_subset(dataframe, selected_ids, train_path, source_path)
            train_manifest = {
                "dataset_id": source_id,
                "source_path": source_path,
                "split_kind": "sample_efficiency_train",
                "fraction": float(fraction),
                "replicate": int(replicate),
                "random_seed": rep_seed,
                "random_source_cells": len(random_cell_ids),
                "selected_source_cells": len(selected_ids),
                "train_path": train_path,
                "source_cell_ids": selected_ids,
                "cell_id_map": mapping,
            }
            train_manifest_path = train_path.parent / "split_manifest.json"
            write_json(train_manifest_path, train_manifest)
            train_manifests.append(train_manifest)

    source_manifest = {
        "dataset_id": source_id,
        "circuit": dataset["circuit"],
        "solver": dataset["solver"],
        "source_path": source_path,
        "split_method": "fixed_test_nested_seeded_train_subsets",
        "random_seed": seed,
        "fixed_test_fraction": test_fraction,
        "random_source_cells": len(random_cell_ids),
        "test_cells": len(test_ids),
        "train_pool_cells": len(train_pool_ids),
        "test_path": test_path,
        "test_source_cell_ids": test_ids,
        "train_pool_source_cell_ids_draw_order": train_pool_ids,
        "test_cell_id_map": test_mapping,
        "train_manifests": train_manifests,
    }
    write_json(source_manifest_path, source_manifest)
    return source_manifest


def prepare_balance_splits(recipe, experiment_root, reuse_existing=True, override_total_fraction=None):
    split_root = split_root_path(recipe, experiment_root)
    source_manifests = load_source_manifests(recipe, experiment_root)
    plateau_info = choose_balance_total_fractions(
        recipe,
        experiment_root,
        source_manifests,
        override_total_fraction=override_total_fraction,
    )
    dataframe_cache = {}
    prepared = []
    for pair_info in balance_pair_infos(recipe):
        left_id = pair_info["left_dataset_id"]
        right_id = pair_info["right_dataset_id"]
        left_manifest = source_manifests[left_id]
        right_manifest = source_manifests[right_id]
        total_fraction = plateau_info["pair_total_fractions"][pair_info["key"]]
        reference_cells = min(
            int(left_manifest["random_source_cells"]),
            int(right_manifest["random_source_cells"]),
        )
        total_cells = int(np.ceil(reference_cells * float(total_fraction)))
        total_cells = min(
            max(total_cells, 2),
            int(left_manifest["train_pool_cells"]) + int(right_manifest["train_pool_cells"]),
        )
        for replicate in replicate_values(recipe):
            for ratio in balance_ratios(recipe):
                paths = balance_training_paths(split_root, pair_info, ratio, replicate)
                manifest_path = balance_manifest_path(split_root, pair_info, ratio, replicate)
                if reuse_existing and all(path.exists() for path in [*paths.values(), manifest_path]):
                    prepared.append(load_json(manifest_path))
                    continue

                left_count, right_count = split_total_by_ratio(total_cells, ratio)
                if left_count > int(left_manifest["train_pool_cells"]):
                    raise ValueError(f"Not enough train-pool cells for {left_id}: requested {left_count}.")
                if right_count > int(right_manifest["train_pool_cells"]):
                    raise ValueError(f"Not enough train-pool cells for {right_id}: requested {right_count}.")

                left_selected = select_balance_source_ids(
                    left_manifest,
                    count=left_count,
                    seed=balance_seed(recipe, pair_info, ratio, replicate, side=0),
                )
                right_selected = select_balance_source_ids(
                    right_manifest,
                    count=right_count,
                    seed=balance_seed(recipe, pair_info, ratio, replicate, side=1),
                )
                left_df = cached_dataframe(dataframe_cache, left_manifest["source_path"])
                right_df = cached_dataframe(dataframe_cache, right_manifest["source_path"])
                left_mapping = write_cell_subset(
                    left_df,
                    left_selected,
                    paths["left"],
                    left_manifest["source_path"],
                )
                right_mapping = write_cell_subset(
                    right_df,
                    right_selected,
                    paths["right"],
                    right_manifest["source_path"],
                )
                manifest = {
                    "split_kind": "mixed_balance_train",
                    "solver": pair_info["solver"],
                    "pair_id": pair_info["pair_id"],
                    "left_circuit": pair_info["left_circuit"],
                    "right_circuit": pair_info["right_circuit"],
                    "left_dataset_id": left_id,
                    "right_dataset_id": right_id,
                    "ratio_left": float(ratio),
                    "ratio_right": float(1.0 - float(ratio)),
                    "replicate": int(replicate),
                    "total_fraction": float(total_fraction),
                    "total_cells": int(total_cells),
                    "left_cells": int(left_count),
                    "right_cells": int(right_count),
                    "left_train_path": paths["left"],
                    "right_train_path": paths["right"],
                    "left_test_path": left_manifest["test_path"],
                    "right_test_path": right_manifest["test_path"],
                    "left_source_cell_ids": left_selected,
                    "right_source_cell_ids": right_selected,
                    "left_cell_id_map": left_mapping,
                    "right_cell_id_map": right_mapping,
                    "plateau_info": plateau_info["pairs"][pair_info["key"]],
                }
                write_json(manifest_path, manifest)
                prepared.append(manifest)
    write_json(split_root / "balance_plateau_selection.json", plateau_info)
    return {
        "split_root": split_root,
        "plateau_info": plateau_info,
        "balance_splits": prepared,
    }


def summarize_balance_splits(prepared):
    return {
        "split_root": prepared.get("split_root"),
        "balance_split_count": len(prepared.get("balance_splits", [])),
        "plateau_strategy": prepared.get("plateau_info", {}).get("strategy"),
        "pair_total_fractions": prepared.get("plateau_info", {}).get("pair_total_fractions", {}),
    }


def prepare_simplex_splits(recipe, experiment_root, reuse_existing=True):
    split_root = split_root_path(recipe, experiment_root)
    source_manifests = load_source_manifests(recipe, experiment_root)
    dataframe_cache = {}
    prepared = []
    budgets = {}
    compositions = simplex_compositions_from_recipe(recipe)

    for solver_info in simplex_solver_infos(recipe):
        solver = solver_info["solver"]
        budget = simplex_total_cell_budget(recipe, experiment_root, source_manifests, solver_info)
        budgets[solver] = budget
        total_cells = int(budget["total_cells"])
        manifests = [source_manifests[item] for item in solver_info["dataset_ids"]]

        for replicate in replicate_values(recipe):
            for composition in compositions:
                paths = simplex_training_paths(split_root, solver_info, composition, replicate)
                manifest_path = simplex_manifest_path(split_root, solver_info, composition, replicate)
                expected = [*paths.values(), manifest_path]
                if reuse_existing and all(path.exists() for path in expected):
                    prepared.append(load_json(manifest_path))
                    continue

                counts = allocate_composition_counts(total_cells, composition)
                selected_ids = []
                mappings = []
                for index, (source_manifest, count) in enumerate(zip(manifests, counts)):
                    if count > int(source_manifest["train_pool_cells"]):
                        raise ValueError(
                            f"Not enough train-pool cells for {source_manifest['dataset_id']}: "
                            f"requested {count}."
                        )
                    selected = select_balance_source_ids(
                        source_manifest,
                        count=count,
                        seed=simplex_seed(recipe, solver, composition, replicate, index),
                    )
                    dataframe = cached_dataframe(dataframe_cache, source_manifest["source_path"])
                    mapping = write_cell_subset(
                        dataframe,
                        selected,
                        paths[source_manifest["dataset_id"]],
                        source_manifest["source_path"],
                    )
                    selected_ids.append(selected)
                    mappings.append(mapping)

                manifest = {
                    "split_kind": "simplex_train",
                    "solver": solver,
                    "circuits": solver_info["circuits"],
                    "dataset_ids": solver_info["dataset_ids"],
                    "composition": list(composition),
                    "composition_id": composition_label(composition),
                    "replicate": int(replicate),
                    "total_cells": total_cells,
                    "cell_counts": list(counts),
                    "train_paths": {key: str(value) for key, value in paths.items()},
                    "test_paths": {
                        item["dataset_id"]: item["test_path"] for item in manifests
                    },
                    "source_cell_ids": dict(zip(solver_info["dataset_ids"], selected_ids)),
                    "cell_id_maps": dict(zip(solver_info["dataset_ids"], mappings)),
                    "total_cell_budget": budget,
                }
                write_json(manifest_path, manifest)
                prepared.append(manifest)

    budget_path = split_root / "simplex_total_cell_budgets.json"
    write_json(
        budget_path,
        {
            "strategy": "maximum_pair_total_cells_per_solver",
            "budgets": budgets,
        },
    )
    return {
        "split_root": split_root,
        "budget_path": budget_path,
        "budgets": budgets,
        "simplex_splits": prepared,
    }


def summarize_simplex_splits(prepared):
    return {
        "split_root": prepared.get("split_root"),
        "simplex_split_count": len(prepared.get("simplex_splits", [])),
        "total_cells_by_solver": {
            solver: details["total_cells"] for solver, details in prepared.get("budgets", {}).items()
        },
        "budget_path": prepared.get("budget_path"),
    }


def simplex_total_cell_budget(recipe, experiment_root, source_manifests, solver_info):
    split_root = split_root_path(recipe, experiment_root)
    pair_budgets = {}
    pair_sources = {}
    plateau_info = None

    for pair_info in balance_pair_infos(recipe):
        if pair_info["solver"] != solver_info["solver"]:
            continue
        pattern_root = (
            split_root
            / "balance"
            / safe_filename(pair_info["solver"])
            / safe_filename(pair_info["pair_id"])
        )
        values = {
            int(load_json(path)["total_cells"])
            for path in pattern_root.glob("*/rep_*/split_manifest.json")
        }
        if len(values) > 1:
            raise ValueError(
                f"Pair split manifests disagree on total_cells for {pair_info['key']}: {sorted(values)}."
            )
        if values:
            pair_budgets[pair_info["pair_id"]] = values.pop()
            pair_sources[pair_info["pair_id"]] = "existing_balance_split_manifests"
            continue

        if plateau_info is None:
            plateau_info = choose_balance_total_fractions(recipe, experiment_root, source_manifests)
        total_fraction = plateau_info["pair_total_fractions"][pair_info["key"]]
        left_manifest = source_manifests[pair_info["left_dataset_id"]]
        right_manifest = source_manifests[pair_info["right_dataset_id"]]
        reference_cells = min(
            int(left_manifest["random_source_cells"]),
            int(right_manifest["random_source_cells"]),
        )
        total_cells = int(np.ceil(reference_cells * float(total_fraction)))
        total_cells = min(
            max(total_cells, 2),
            int(left_manifest["train_pool_cells"]) + int(right_manifest["train_pool_cells"]),
        )
        pair_budgets[pair_info["pair_id"]] = total_cells
        pair_sources[pair_info["pair_id"]] = "sample_efficiency_plateau_fallback"

    if not pair_budgets:
        raise ValueError(f"No pair cell budgets are available for solver {solver_info['solver']!r}.")
    return {
        "solver": solver_info["solver"],
        "strategy": "maximum_pair_total_cells",
        "total_cells": max(pair_budgets.values()),
        "pair_total_cells": pair_budgets,
        "pair_budget_sources": pair_sources,
    }


def run_sample_efficiency_cell(
    recipe,
    experiment_root,
    forecaster_id,
    dataset_id_value,
    fraction,
    replicate,
    visualization=None,
):
    forecaster = forecaster_by_id(recipe, forecaster_id)
    dataset = dataset_by_id(recipe, dataset_id_value)
    split_root = split_root_path(recipe, experiment_root)
    train_path = sample_train_path(split_root, dataset_id_value, fraction, replicate)
    test_path = source_fixed_test_path(split_root, dataset_id_value)
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Sample split files are missing. Run --prepare-sample-splits-only first.")

    model_config = resolved_model_config(
        recipe,
        forecaster,
        {dataset_id_value: train_path, f"{dataset_id_value}_test": test_path},
    )
    output_root = (
        experiment_root
        / DEFAULT_SAMPLE_ROOT
        / "runs"
        / safe_filename(forecaster_id)
        / safe_filename(dataset_id_value)
        / fraction_label(fraction)
        / replicate_label(replicate)
    )
    label = f"{safe_filename(dataset_id_value)}_{fraction_label(fraction)}_{replicate_label(replicate)}"
    result = cross_test_forecaster(
        model=model_config,
        training_data=train_path,
        test_data=test_path,
        output_root=output_root,
        label=label,
        random_state=run_seed(recipe, forecaster_id, dataset_id_value, fraction, replicate),
        include_main_periodic=include_main_periodic(recipe),
        include_test_periodic=include_test_periodic(recipe),
        visualization=use_visualization(recipe, visualization),
    )
    train_manifest = load_json(train_path.parent / "split_manifest.json")
    source_manifest = load_json(split_root / "sources" / safe_filename(dataset_id_value) / "source_manifest.json")
    record = result_record_base(result)
    record.update(
        {
            "mode": "sample_efficiency",
            "forecaster_id": forecaster_id,
            "dataset_id": dataset_id_value,
            "circuit": dataset["circuit"],
            "solver": dataset["solver"],
            "fraction": float(fraction),
            "replicate": int(replicate),
            "train_cells": int(train_manifest["selected_source_cells"]),
            "test_cells": int(source_manifest["test_cells"]),
            "train_path": train_path,
            "test_path": test_path,
        }
    )
    record_path = sample_record_path(experiment_root, forecaster_id, dataset_id_value, fraction, replicate)
    write_json(record_path, record)
    return record


def run_balance_cell(
    recipe,
    experiment_root,
    forecaster_id,
    solver,
    pair_id,
    ratio,
    replicate,
    visualization=None,
):
    forecaster = forecaster_by_id(recipe, forecaster_id)
    pair_info = balance_pair_by_id(recipe, solver, pair_id)
    split_root = split_root_path(recipe, experiment_root)
    manifest_path = balance_manifest_path(split_root, pair_info, ratio, replicate)
    if not manifest_path.exists():
        raise FileNotFoundError("Balance split files are missing. Run --prepare-balance-splits-only first.")
    manifest = load_json(manifest_path)
    train_paths = [Path(manifest["left_train_path"]), Path(manifest["right_train_path"])]
    test_targets = [
        (pair_info["left_dataset_id"], pair_info["left_circuit"], Path(manifest["left_test_path"])),
        (pair_info["right_dataset_id"], pair_info["right_circuit"], Path(manifest["right_test_path"])),
    ]
    model_config = resolved_model_config(
        recipe,
        forecaster,
        {
            pair_info["left_dataset_id"]: train_paths[0],
            pair_info["right_dataset_id"]: train_paths[1],
            f"{pair_info['left_dataset_id']}_test": test_targets[0][2],
            f"{pair_info['right_dataset_id']}_test": test_targets[1][2],
        },
    )
    output_root = (
        experiment_root
        / DEFAULT_BALANCE_ROOT
        / "runs"
        / safe_filename(forecaster_id)
        / safe_filename(solver)
        / safe_filename(pair_id)
        / ratio_label(ratio)
        / replicate_label(replicate)
    )
    records = []
    trained_model = None
    for target_index, (test_dataset_id, test_circuit, test_path) in enumerate(test_targets):
        label = (
            f"{safe_filename(pair_id)}_{ratio_label(ratio)}_{replicate_label(replicate)}"
            f"_to_{safe_filename(test_dataset_id)}"
        )
        if trained_model is None:
            result = cross_test_forecaster(
                model=model_config,
                training_data=train_paths,
                test_data=test_path,
                output_root=output_root,
                label=label,
                random_state=run_seed(recipe, forecaster_id, solver, pair_id, ratio, replicate),
                include_main_periodic=include_main_periodic(recipe),
                include_test_periodic=include_test_periodic(recipe),
                visualization=use_visualization(recipe, visualization),
            )
            trained_model = result["model_path"]
        else:
            result = cross_test_forecaster(
                trained_model=trained_model,
                test_data=test_path,
                output_root=output_root,
                label=label,
                random_state=run_seed(recipe, forecaster_id, solver, pair_id, ratio, replicate, target_index),
                include_test_periodic=include_test_periodic(recipe),
                visualization=use_visualization(recipe, visualization),
            )
        record = result_record_base(result)
        record.update(
            {
                "mode": "balance",
                "forecaster_id": forecaster_id,
                "solver": solver,
                "pair_id": pair_id,
                "left_circuit": pair_info["left_circuit"],
                "right_circuit": pair_info["right_circuit"],
                "left_dataset_id": pair_info["left_dataset_id"],
                "right_dataset_id": pair_info["right_dataset_id"],
                "test_dataset_id": test_dataset_id,
                "test_circuit": test_circuit,
                "ratio_left": float(ratio),
                "ratio_right": float(1.0 - float(ratio)),
                "replicate": int(replicate),
                "total_cells": int(manifest["total_cells"]),
                "left_cells": int(manifest["left_cells"]),
                "right_cells": int(manifest["right_cells"]),
                "train_paths": train_paths,
                "test_path": test_path,
                "model_path": result["model_path"],
            }
        )
        records.append(record)
    record_path = balance_record_path(experiment_root, forecaster_id, solver, pair_id, ratio, replicate)
    write_json(record_path, {"records": records})
    return {"records": records, "record_path": record_path}


def run_simplex_cell(
    recipe,
    experiment_root,
    forecaster_id,
    solver,
    composition,
    replicate,
    visualization=None,
):
    forecaster = forecaster_by_id(recipe, forecaster_id)
    solver_info = simplex_solver_by_id(recipe, solver)
    composition = validate_simplex_composition(recipe, composition)
    split_root = split_root_path(recipe, experiment_root)
    manifest_path = simplex_manifest_path(split_root, solver_info, composition, replicate)
    if not manifest_path.exists():
        raise FileNotFoundError("Simplex split files are missing. Run --prepare-simplex-splits-only first.")
    manifest = load_json(manifest_path)
    train_paths = [Path(manifest["train_paths"][item]) for item in solver_info["dataset_ids"]]
    test_targets = [
        (
            dataset_id_value,
            circuit,
            Path(manifest["test_paths"][dataset_id_value]),
        )
        for dataset_id_value, circuit in zip(solver_info["dataset_ids"], solver_info["circuits"])
    ]
    model_paths = {
        dataset_id_value: train_path
        for dataset_id_value, train_path in zip(solver_info["dataset_ids"], train_paths)
    }
    model_paths.update(
        {f"{dataset_id_value}_test": test_path for dataset_id_value, _, test_path in test_targets}
    )
    model_config = resolved_model_config(recipe, forecaster, model_paths)
    composition_id = composition_label(composition)
    output_root = (
        experiment_root
        / DEFAULT_SIMPLEX_ROOT
        / "runs"
        / safe_filename(forecaster_id)
        / safe_filename(solver)
        / composition_id
        / replicate_label(replicate)
    )
    records = []
    trained_model = None
    for target_index, (test_dataset_id, test_circuit, test_path) in enumerate(test_targets):
        label = (
            f"{composition_id}_{replicate_label(replicate)}"
            f"_to_{safe_filename(test_dataset_id)}"
        )
        if trained_model is None:
            result = cross_test_forecaster(
                model=model_config,
                training_data=train_paths,
                test_data=test_path,
                output_root=output_root,
                label=label,
                random_state=run_seed(
                    recipe, forecaster_id, solver, composition_id, replicate
                ),
                include_main_periodic=include_main_periodic(recipe),
                include_test_periodic=include_test_periodic(recipe),
                visualization=use_visualization(recipe, visualization),
            )
            trained_model = result["model_path"]
        else:
            result = cross_test_forecaster(
                trained_model=trained_model,
                test_data=test_path,
                output_root=output_root,
                label=label,
                random_state=run_seed(
                    recipe, forecaster_id, solver, composition_id, replicate, target_index
                ),
                include_test_periodic=include_test_periodic(recipe),
                visualization=use_visualization(recipe, visualization),
            )
        record = result_record_base(result)
        record.update(
            {
                "mode": "simplex",
                "forecaster_id": forecaster_id,
                "solver": solver,
                "circuits": solver_info["circuits"],
                "dataset_ids": solver_info["dataset_ids"],
                "composition_id": composition_id,
                "test_dataset_id": test_dataset_id,
                "test_circuit": test_circuit,
                "fraction_1": float(composition[0]),
                "fraction_2": float(composition[1]),
                "fraction_3": float(composition[2]),
                "replicate": int(replicate),
                "total_cells": int(manifest["total_cells"]),
                "cells_1": int(manifest["cell_counts"][0]),
                "cells_2": int(manifest["cell_counts"][1]),
                "cells_3": int(manifest["cell_counts"][2]),
                "train_paths": train_paths,
                "test_path": test_path,
                "model_path": result["model_path"],
            }
        )
        records.append(record)
    record_path = simplex_record_path(
        experiment_root, forecaster_id, solver, composition, replicate
    )
    write_json(record_path, {"records": records})
    return {"records": records, "record_path": record_path}


def aggregate_training_data_results(recipe, experiment_root):
    sample_summary = aggregate_sample_results(recipe, experiment_root)
    balance_summary = aggregate_balance_results(recipe, experiment_root)
    simplex_summary = aggregate_simplex_results(recipe, experiment_root)
    return {
        "sample_efficiency": sample_summary,
        "balance": balance_summary,
        "simplex": simplex_summary,
    }


def aggregate_sample_results(recipe, experiment_root):
    records = []
    record_root = experiment_root / DEFAULT_SAMPLE_ROOT / "records"
    for path in sorted(record_root.glob("**/*.json")):
        records.append(load_json(path))
    summary_path = experiment_root / DEFAULT_SAMPLE_ROOT / "sample_efficiency_summary.csv"
    aggregate_path = experiment_root / DEFAULT_SAMPLE_ROOT / "sample_efficiency_aggregate.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        pd.DataFrame(columns=SAMPLE_SUMMARY_COLUMNS).to_csv(summary_path, index=False)
        pd.DataFrame(columns=SAMPLE_AGGREGATE_COLUMNS).to_csv(aggregate_path, index=False)
        return {"summary": summary_path, "aggregate": aggregate_path, "records": 0}

    frame = pd.DataFrame(records)
    frame = frame.sort_values(["forecaster_id", "dataset_id", "fraction", "replicate"])
    frame.to_csv(summary_path, index=False)
    aggregate = (
        frame.groupby(["forecaster_id", "dataset_id", "circuit", "solver", "fraction"], dropna=False)
        .agg(
            mean_rmse_mean=("mean_rmse", "mean"),
            mean_rmse_std=("mean_rmse", "std"),
            mean_rmse_count=("mean_rmse", "count"),
            train_cells_mean=("train_cells", "mean"),
            test_cells_mean=("test_cells", "mean"),
        )
        .reset_index()
    )
    aggregate.to_csv(aggregate_path, index=False)
    return {"summary": summary_path, "aggregate": aggregate_path, "records": len(records)}


def aggregate_balance_results(recipe, experiment_root):
    records = []
    record_root = experiment_root / DEFAULT_BALANCE_ROOT / "records"
    for path in sorted(record_root.glob("**/*.json")):
        payload = load_json(path)
        records.extend(payload.get("records", []))
    summary_path = experiment_root / DEFAULT_BALANCE_ROOT / "balance_summary.csv"
    aggregate_path = experiment_root / DEFAULT_BALANCE_ROOT / "balance_aggregate.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        pd.DataFrame(columns=BALANCE_SUMMARY_COLUMNS).to_csv(summary_path, index=False)
        pd.DataFrame(columns=BALANCE_AGGREGATE_COLUMNS).to_csv(aggregate_path, index=False)
        return {"summary": summary_path, "aggregate": aggregate_path, "records": 0}

    frame = pd.DataFrame(records)
    frame = frame.sort_values(
        ["forecaster_id", "solver", "pair_id", "ratio_left", "replicate", "test_dataset_id"]
    )
    frame.to_csv(summary_path, index=False)
    aggregate = (
        frame.groupby(
            [
                "forecaster_id",
                "solver",
                "pair_id",
                "left_circuit",
                "right_circuit",
                "test_dataset_id",
                "test_circuit",
                "ratio_left",
                "ratio_right",
            ],
            dropna=False,
        )
        .agg(
            mean_rmse_mean=("mean_rmse", "mean"),
            mean_rmse_std=("mean_rmse", "std"),
            mean_rmse_count=("mean_rmse", "count"),
            total_cells_mean=("total_cells", "mean"),
            left_cells_mean=("left_cells", "mean"),
            right_cells_mean=("right_cells", "mean"),
        )
        .reset_index()
    )
    aggregate.to_csv(aggregate_path, index=False)
    return {"summary": summary_path, "aggregate": aggregate_path, "records": len(records)}


def aggregate_simplex_results(recipe, experiment_root):
    records = []
    record_root = experiment_root / DEFAULT_SIMPLEX_ROOT / "records"
    for path in sorted(record_root.glob("**/*.json")):
        payload = load_json(path)
        records.extend(payload.get("records", []))
    summary_path = experiment_root / DEFAULT_SIMPLEX_ROOT / "simplex_summary.csv"
    aggregate_path = experiment_root / DEFAULT_SIMPLEX_ROOT / "simplex_aggregate.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        pd.DataFrame(columns=SIMPLEX_SUMMARY_COLUMNS).to_csv(summary_path, index=False)
        pd.DataFrame(columns=SIMPLEX_AGGREGATE_COLUMNS).to_csv(aggregate_path, index=False)
        return {"summary": summary_path, "aggregate": aggregate_path, "records": 0}

    frame = pd.DataFrame(records)
    frame = frame.sort_values(
        ["forecaster_id", "solver", "composition_id", "replicate", "test_dataset_id"]
    )
    frame.to_csv(summary_path, index=False)
    aggregate = (
        frame.groupby(
            [
                "forecaster_id",
                "solver",
                "composition_id",
                "test_dataset_id",
                "test_circuit",
                "fraction_1",
                "fraction_2",
                "fraction_3",
            ],
            dropna=False,
        )
        .agg(
            mean_rmse_mean=("mean_rmse", "mean"),
            mean_rmse_std=("mean_rmse", "std"),
            mean_rmse_count=("mean_rmse", "count"),
            total_cells_mean=("total_cells", "mean"),
            cells_1_mean=("cells_1", "mean"),
            cells_2_mean=("cells_2", "mean"),
            cells_3_mean=("cells_3", "mean"),
        )
        .reset_index()
    )
    aggregate.to_csv(aggregate_path, index=False)
    return {"summary": summary_path, "aggregate": aggregate_path, "records": len(records)}


def plot_training_data_results(recipe, experiment_root):
    sample_plots = plot_sample_efficiency_results(recipe, experiment_root)
    balance_plots = plot_balance_results(recipe, experiment_root)
    simplex_plots = plot_simplex_results(recipe, experiment_root)
    return {
        "sample_efficiency": sample_plots,
        "balance": balance_plots,
        "simplex": simplex_plots,
    }


def plot_sample_efficiency_results(recipe, experiment_root):
    aggregate_path = experiment_root / DEFAULT_SAMPLE_ROOT / "sample_efficiency_aggregate.csv"
    if not aggregate_path.exists():
        raise FileNotFoundError(f"Missing sample-efficiency aggregate: {aggregate_path}")
    aggregate = read_csv_allow_empty(aggregate_path, SAMPLE_AGGREGATE_COLUMNS)
    if aggregate.empty:
        return []

    plot_root = experiment_root / DEFAULT_SAMPLE_ROOT / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for _, forecaster in enumerate(recipe["forecasters"]):
        forecaster_id = forecaster["id"]
        for dataset in recipe["datasets"]:
            dataset_id_value = dataset_id(dataset)
            subset = aggregate[
                (aggregate["forecaster_id"] == forecaster_id)
                & (aggregate["dataset_id"] == dataset_id_value)
            ].sort_values("fraction")
            if subset.empty:
                continue
            title = f"{forecaster_id}: {dataset_id_value}"
            prefix = f"{safe_filename(forecaster_id)}_{safe_filename(dataset_id_value)}_sample_efficiency"
            for suffix in ("png", "svg"):
                output_path = plot_root / f"{prefix}.{suffix}"
                plot_sample_efficiency_curve(subset, output_path, title)
                outputs.append(output_path)
    return outputs


def plot_balance_results(recipe, experiment_root):
    aggregate_path = experiment_root / DEFAULT_BALANCE_ROOT / "balance_aggregate.csv"
    if not aggregate_path.exists():
        raise FileNotFoundError(f"Missing balance aggregate: {aggregate_path}")
    aggregate = read_csv_allow_empty(aggregate_path, BALANCE_AGGREGATE_COLUMNS)
    if aggregate.empty:
        return []

    plot_root = experiment_root / DEFAULT_BALANCE_ROOT / "plots"
    pair_root = plot_root / "pair_matrices"
    target_root = plot_root / "target_matrices"
    pair_root.mkdir(parents=True, exist_ok=True)
    target_root.mkdir(parents=True, exist_ok=True)
    scale = global_value_scale(aggregate["mean_rmse_mean"].to_numpy(dtype=float))
    outputs = []
    for forecaster in recipe["forecasters"]:
        forecaster_id = forecaster["id"]
        for pair_info in balance_pair_infos(recipe):
            subset = aggregate[
                (aggregate["forecaster_id"] == forecaster_id)
                & (aggregate["solver"] == pair_info["solver"])
                & (aggregate["pair_id"] == pair_info["pair_id"])
            ]
            if subset.empty:
                continue
            matrix = balance_pair_matrix(subset, pair_info)
            title = f"{forecaster_id}: {pair_info['solver']} {pair_info['pair_id']}"
            prefix = (
                f"{safe_filename(forecaster_id)}_{safe_filename(pair_info['solver'])}_"
                f"{safe_filename(pair_info['pair_id'])}"
            )
            for suffix in ("png", "svg"):
                output_path = pair_root / f"{prefix}.{suffix}"
                plot_heatmap_matrix(matrix, output_path, title, scale=scale)
                outputs.append(output_path)
            for test_id in matrix.index:
                target_matrix = matrix.loc[[test_id], :]
                target_prefix = f"{prefix}_test_{safe_filename(test_id)}"
                target_title = f"{title} tested on {test_id}"
                for suffix in ("png", "svg"):
                    output_path = target_root / f"{target_prefix}.{suffix}"
                    plot_heatmap_matrix(target_matrix, output_path, target_title, scale=scale)
                    outputs.append(output_path)
    return outputs


def plot_simplex_results(recipe, experiment_root):
    aggregate_path = experiment_root / DEFAULT_SIMPLEX_ROOT / "simplex_aggregate.csv"
    if not aggregate_path.exists():
        raise FileNotFoundError(f"Missing simplex aggregate: {aggregate_path}")
    aggregate = read_csv_allow_empty(aggregate_path, SIMPLEX_AGGREGATE_COLUMNS)
    if aggregate.empty:
        return []

    plot_root = experiment_root / DEFAULT_SIMPLEX_ROOT / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    scale = global_value_scale(aggregate["mean_rmse_mean"].to_numpy(dtype=float))
    outputs = []
    for forecaster in recipe["forecasters"]:
        forecaster_id = forecaster["id"]
        for solver_info in simplex_solver_infos(recipe):
            subset = aggregate[
                (aggregate["forecaster_id"] == forecaster_id)
                & (aggregate["solver"] == solver_info["solver"])
            ]
            if subset.empty:
                continue
            prefix = (
                f"{safe_filename(forecaster_id)}_{safe_filename(solver_info['solver'])}"
                "_simplex"
            )
            title = f"{forecaster_id}: {solver_info['solver']}"
            for suffix in ("png", "svg"):
                output_path = plot_root / f"{prefix}.{suffix}"
                plot_simplex_panels(subset, solver_info, output_path, title, scale)
                outputs.append(output_path)
    return outputs


def plot_simplex_panels(frame, solver_info, output_path, title, scale):
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(Path(output_path).parent / ".matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.tri import Triangulation

    circuits = solver_info["circuits"]
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=scale["vmin"], vmax=scale["vmax"])
    triangle_x = [0.0, 1.0, 0.5, 0.0]
    triangle_y = [0.0, 0.0, np.sqrt(3.0) / 2.0, 0.0]

    for axis, test_circuit in zip(axes, circuits):
        target = frame[frame["test_circuit"] == test_circuit].sort_values(
            ["fraction_1", "fraction_2", "fraction_3"]
        )
        fraction_2 = target["fraction_2"].to_numpy(dtype=float)
        fraction_3 = target["fraction_3"].to_numpy(dtype=float)
        x = fraction_2 + 0.5 * fraction_3
        y = (np.sqrt(3.0) / 2.0) * fraction_3
        values = target["mean_rmse_mean"].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
        x, y, values = x[finite], y[finite], values[finite]

        if len(values) >= 3:
            try:
                triangulation = Triangulation(x, y)
                axis.tripcolor(
                    triangulation,
                    values,
                    shading="gouraud",
                    cmap=cmap,
                    norm=norm,
                )
            except (RuntimeError, ValueError):
                axis.scatter(x, y, c=values, cmap=cmap, norm=norm, s=55)
        elif len(values):
            axis.scatter(x, y, c=values, cmap=cmap, norm=norm, s=55)
        axis.scatter(x, y, c="black", s=5, alpha=0.45, zorder=3)
        axis.plot(triangle_x, triangle_y, color="#333333", linewidth=1.0)
        axis.text(-0.03, -0.045, circuits[0], ha="right", va="top", fontsize=9)
        axis.text(1.03, -0.045, circuits[1], ha="left", va="top", fontsize=9)
        axis.text(0.5, triangle_y[2] + 0.045, circuits[2], ha="center", va="bottom", fontsize=9)
        axis.set_title(f"Test: {test_circuit}", fontsize=10)
        axis.set_xlim(-0.12, 1.12)
        axis.set_ylim(-0.09, triangle_y[2] + 0.1)
        axis.set_aspect("equal")
        axis.axis("off")

    figure.suptitle(title, fontsize=12)
    colorbar = figure.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        shrink=0.78,
        pad=0.02,
    )
    colorbar.set_label("Mean RMSE")
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_sample_efficiency_curve(frame, output_path, title):
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(Path(output_path).parent.parent / ".matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    x = frame["fraction"].to_numpy(dtype=float) * 100.0
    y = frame["mean_rmse_mean"].to_numpy(dtype=float)
    yerr = frame["mean_rmse_std"].fillna(0.0).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    ax.errorbar(x, y, yerr=yerr, fmt="o-", linewidth=1.6, markersize=4.5, capsize=3)
    ax.set_title(title)
    ax.set_xlabel("Training cells (% of random-stim source cells)")
    ax.set_ylabel("Mean RMSE")
    ax.set_xticks(x)
    ax.grid(True, linewidth=0.5, alpha=0.35)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_heatmap_matrix(matrix, output_path, title, scale):
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(Path(output_path).parent.parent / ".matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    values = matrix.astype(float).to_numpy()
    masked = np.ma.masked_invalid(values)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#f1f1f1")
    fig_width = max(5.8, 0.62 * len(matrix.columns) + 2.2)
    fig_height = max(2.4, 0.58 * len(matrix.index) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    image = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=scale["vmin"], vmax=scale["vmax"])
    ax.set_title(title)
    ax.set_xlabel("Left-circuit training share")
    ax.set_ylabel("Test dataset")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    midpoint = scale["midpoint"]
    for row_index, row_name in enumerate(matrix.index):
        for col_index, column_name in enumerate(matrix.columns):
            value = matrix.loc[row_name, column_name]
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
                fontsize=7,
            )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.9)
    colorbar.set_label("Mean RMSE")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def balance_pair_matrix(frame, pair_info):
    columns = [ratio_column_label(ratio, pair_info) for ratio in balance_ratios_from_frame(frame)]
    test_ids = [pair_info["left_dataset_id"], pair_info["right_dataset_id"]]
    matrix = pd.DataFrame(index=test_ids, columns=columns, dtype=float)
    for _, row in frame.iterrows():
        column = ratio_column_label(float(row["ratio_left"]), pair_info)
        matrix.loc[str(row["test_dataset_id"]), column] = float(row["mean_rmse_mean"])
    return matrix


def balance_ratios_from_frame(frame):
    return sorted(float(value) for value in frame["ratio_left"].dropna().unique())


def global_value_scale(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"vmin": 0.0, "vmax": 1.0, "midpoint": 0.5}
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if vmin == vmax:
        pad = max(1.0, abs(vmin) * 0.05)
        vmin -= pad
        vmax += pad
    return {
        "vmin": vmin,
        "vmax": vmax,
        "midpoint": float(np.nanmedian(finite)),
    }


def choose_balance_total_fractions(recipe, experiment_root, source_manifests, override_total_fraction=None):
    fallback = float(
        override_total_fraction
        if override_total_fraction is not None
        else recipe.get("balance", {}).get(
            "default_total_fraction",
            recipe.get("plateau", {}).get("fallback_fraction", 0.5),
        )
    )
    fallback = clamp_fraction(fallback)
    tolerance = float(recipe.get("plateau", {}).get("relative_to_best_tolerance", 0.05))
    sample_aggregate_path = experiment_root / DEFAULT_SAMPLE_ROOT / "sample_efficiency_aggregate.csv"
    selected_by_forecaster_dataset = {}
    if sample_aggregate_path.exists():
        aggregate = read_csv_allow_empty(sample_aggregate_path, SAMPLE_AGGREGATE_COLUMNS)
        if not aggregate.empty:
            for (forecaster_id, dataset_id_value), group in aggregate.groupby(["forecaster_id", "dataset_id"]):
                selected_by_forecaster_dataset[(forecaster_id, dataset_id_value)] = select_plateau_fraction(
                    group,
                    tolerance=tolerance,
                    fallback=fallback,
                )

    pair_total_fractions = {}
    pair_details = {}
    for pair_info in balance_pair_infos(recipe):
        candidate_fractions = []
        for forecaster in recipe["forecasters"]:
            for dataset_id_value in (pair_info["left_dataset_id"], pair_info["right_dataset_id"]):
                candidate_fractions.append(
                    selected_by_forecaster_dataset.get((forecaster["id"], dataset_id_value), fallback)
                )
        selected_fraction = clamp_fraction(max(candidate_fractions) if candidate_fractions else fallback)
        key = pair_info["key"]
        pair_total_fractions[key] = selected_fraction
        pair_details[key] = {
            "solver": pair_info["solver"],
            "pair_id": pair_info["pair_id"],
            "left_dataset_id": pair_info["left_dataset_id"],
            "right_dataset_id": pair_info["right_dataset_id"],
            "selected_fraction": selected_fraction,
            "fallback_fraction": fallback,
            "relative_to_best_tolerance": tolerance,
            "candidate_fractions": candidate_fractions,
        }
    return {
        "strategy": recipe.get("balance", {}).get("total_size_strategy", "global_pair_max_plateau"),
        "pair_total_fractions": pair_total_fractions,
        "pairs": pair_details,
        "selected_by_forecaster_dataset": {
            f"{forecaster_id}::{dataset_id_value}": value
            for (forecaster_id, dataset_id_value), value in selected_by_forecaster_dataset.items()
        },
        "source_cells": {
            source_id: {
                "random_source_cells": manifest["random_source_cells"],
                "train_pool_cells": manifest["train_pool_cells"],
                "test_cells": manifest["test_cells"],
            }
            for source_id, manifest in source_manifests.items()
        },
    }


def select_plateau_fraction(group, tolerance, fallback):
    group = group.sort_values("fraction")
    finite = group[np.isfinite(group["mean_rmse_mean"].astype(float))]
    if finite.empty:
        return fallback
    best = float(finite["mean_rmse_mean"].min())
    if best <= 0:
        threshold = best + abs(best) * tolerance
    else:
        threshold = best * (1.0 + tolerance)
    eligible = finite[finite["mean_rmse_mean"] <= threshold]
    if eligible.empty:
        return fallback
    return float(eligible.sort_values("fraction").iloc[0]["fraction"])


def result_record_base(result):
    test_metrics = result["metrics"]["test"]
    training_holdout = result["metrics"].get("training_holdout")
    return {
        "mean_rmse": test_metrics.get("rmse"),
        "finite_windows": test_metrics.get("finite_windows"),
        "invalid_windows": test_metrics.get("invalid_windows"),
        "training_holdout_rmse": training_holdout.get("rmse") if isinstance(training_holdout, Mapping) else None,
        "output_dir": result["output_dir"],
        "model_path": result["model_path"],
        "test_performance_path": result["test_performance_path"],
        "training_holdout_performance_path": result["training_holdout_performance_path"],
    }


def resolved_model_config(recipe, forecaster, dataset_paths):
    reporter_species = recipe.get("reporter_species", DEFAULT_REPORTER_SPECIES)
    return reporter_only_model_config(
        copy.deepcopy(forecaster["model"]),
        dataset_paths,
        reporter_species=reporter_species,
        force_reporter_only=False,
    )


def write_cell_subset(dataframe, source_cell_ids, output_path, source_path):
    selected = set(source_cell_ids)
    frame = dataframe[dataframe["cell_id"].isin(selected)].copy()
    if frame.empty:
        raise ValueError(f"No rows selected for {output_path}.")
    mapping = _remap_cell_ids(frame)
    _write_split_dataset(frame, output_path, Path(source_path), len(mapping))
    return mapping


def cached_dataframe(cache, source_path):
    source_path = str(source_path)
    if source_path not in cache:
        cache[source_path] = pd.read_parquet(source_path)
    return cache[source_path]


def load_source_manifests(recipe, experiment_root):
    split_root = split_root_path(recipe, experiment_root)
    manifests = {}
    for dataset in recipe["datasets"]:
        source_id = dataset_id(dataset)
        manifest_path = split_root / "sources" / safe_filename(source_id) / "source_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing source split manifest: {manifest_path}")
        manifests[source_id] = load_json(manifest_path)
    return manifests


def balance_pair_infos(recipe):
    datasets = list(recipe["datasets"])
    circuits = list(recipe.get("balance", {}).get("circuits", DEFAULT_CIRCUITS))
    by_solver_circuit = {}
    solvers = []
    for dataset in datasets:
        solver = str(dataset["solver"])
        circuit = str(dataset["circuit"])
        if solver not in solvers:
            solvers.append(solver)
        by_solver_circuit[(solver, circuit)] = dataset_id(dataset)

    pairs = []
    for solver in solvers:
        for left_index in range(len(circuits)):
            for right_index in range(left_index + 1, len(circuits)):
                left_circuit = circuits[left_index]
                right_circuit = circuits[right_index]
                if (solver, left_circuit) not in by_solver_circuit or (solver, right_circuit) not in by_solver_circuit:
                    continue
                pair_id = f"{left_circuit}_vs_{right_circuit}"
                key = f"{solver}::{pair_id}"
                pairs.append(
                    {
                        "key": key,
                        "solver": solver,
                        "pair_id": pair_id,
                        "left_circuit": left_circuit,
                        "right_circuit": right_circuit,
                        "left_dataset_id": by_solver_circuit[(solver, left_circuit)],
                        "right_dataset_id": by_solver_circuit[(solver, right_circuit)],
                    }
                )
    return pairs


def balance_pair_by_id(recipe, solver, pair_id):
    for pair_info in balance_pair_infos(recipe):
        if pair_info["solver"] == solver and pair_info["pair_id"] == pair_id:
            return pair_info
    available = [f"{item['solver']}::{item['pair_id']}" for item in balance_pair_infos(recipe)]
    raise ValueError(f"Unknown balance pair {solver!r}::{pair_id!r}. Available: {available}")


def simplex_solver_infos(recipe):
    circuits = list(
        recipe.get("simplex", {}).get(
            "circuits",
            recipe.get("balance", {}).get("circuits", DEFAULT_CIRCUITS),
        )
    )
    by_solver_circuit = {}
    solvers = []
    for dataset in recipe["datasets"]:
        solver = str(dataset["solver"])
        circuit = str(dataset["circuit"])
        if solver not in solvers:
            solvers.append(solver)
        by_solver_circuit[(solver, circuit)] = dataset_id(dataset)

    results = []
    for solver in solvers:
        if not all((solver, circuit) in by_solver_circuit for circuit in circuits):
            continue
        results.append(
            {
                "solver": solver,
                "circuits": circuits,
                "dataset_ids": [by_solver_circuit[(solver, circuit)] for circuit in circuits],
            }
        )
    return results


def simplex_solver_by_id(recipe, solver):
    for solver_info in simplex_solver_infos(recipe):
        if solver_info["solver"] == solver:
            return solver_info
    available = [item["solver"] for item in simplex_solver_infos(recipe)]
    raise ValueError(f"Unknown simplex solver {solver!r}. Available: {available}")


def simplex_compositions_from_recipe(recipe):
    config = recipe.get("simplex", {})
    return simplex_compositions(
        step=float(config.get("step", 0.1)),
        minimum_fraction=float(config.get("minimum_fraction", config.get("step", 0.1))),
    )


def simplex_compositions(step=0.1, minimum_fraction=0.1):
    if step <= 0 or step >= 1:
        raise ValueError("Simplex step must be in the range (0, 1).")
    if minimum_fraction <= 0 or minimum_fraction >= 1 / 3:
        raise ValueError("Simplex minimum_fraction must be in the range (0, 1/3).")
    units = int(round(1.0 / step))
    minimum_units = int(round(minimum_fraction / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("Simplex step must divide 1.0 exactly.")
    if minimum_units < 1 or not np.isclose(minimum_units * step, minimum_fraction):
        raise ValueError("Simplex minimum_fraction must be a positive multiple of step.")

    compositions = []
    for first in range(minimum_units, units + 1):
        for second in range(minimum_units, units + 1):
            third = units - first - second
            if third < minimum_units:
                continue
            compositions.append((first / units, second / units, third / units))
    if not compositions:
        raise ValueError("Simplex settings do not produce any valid three-circuit compositions.")
    return compositions


def parse_composition(value):
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    else:
        parts = list(value)
    if len(parts) != 3:
        raise ValueError("Composition must contain three comma-separated fractions.")
    return tuple(float(part) for part in parts)


def validate_simplex_composition(recipe, composition):
    composition = parse_composition(composition)
    for allowed in simplex_compositions_from_recipe(recipe):
        if np.allclose(composition, allowed, rtol=0.0, atol=1e-9):
            return allowed
    raise ValueError(
        f"Composition {composition} is not on the configured simplex lattice."
    )


def allocate_composition_counts(total_cells, composition):
    total_cells = int(total_cells)
    fractions = np.asarray(composition, dtype=float)
    if total_cells < len(fractions):
        raise ValueError("Simplex total_cells must provide at least one cell per circuit.")
    if np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("Simplex composition fractions must be positive and sum to 1.")

    raw = fractions * total_cells
    counts = np.floor(raw).astype(int)
    counts[counts < 1] = 1
    while counts.sum() < total_cells:
        order = np.argsort(-(raw - counts))
        for index in order:
            if counts.sum() == total_cells:
                break
            counts[index] += 1
    while counts.sum() > total_cells:
        candidates = [index for index in np.argsort(-(counts - raw)) if counts[index] > 1]
        if not candidates:
            raise ValueError("Could not allocate simplex cells while retaining every circuit.")
        counts[candidates[0]] -= 1
    return tuple(int(value) for value in counts)


def select_balance_source_ids(source_manifest, count, seed):
    pool = np.array(source_manifest["train_pool_source_cell_ids_draw_order"], dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(pool)
    return pool[: int(count)].tolist()


def split_total_by_ratio(total_cells, ratio):
    total_cells = int(total_cells)
    left_count = int(round(total_cells * float(ratio)))
    left_count = min(max(left_count, 1), total_cells - 1)
    right_count = total_cells - left_count
    return left_count, right_count


def forecaster_by_id(recipe, forecaster_id):
    for forecaster in recipe["forecasters"]:
        if forecaster["id"] == forecaster_id:
            return forecaster
    raise ValueError(f"Unknown forecaster {forecaster_id!r}.")


def dataset_by_id(recipe, dataset_id_value):
    for dataset in recipe["datasets"]:
        if dataset_id(dataset) == dataset_id_value:
            return dataset
    raise ValueError(f"Unknown dataset {dataset_id_value!r}.")


def split_root_path(recipe, experiment_root):
    return _resolve_path(recipe.get("split", {}).get("root", DEFAULT_SPLIT_ROOT), base=experiment_root)


def sample_fractions(recipe):
    return [float(value) for value in recipe["sample_efficiency"]["fractions"]]


def balance_ratios(recipe):
    return [float(value) for value in recipe["balance"]["ratios"]]


def replicate_values(recipe):
    count = int(recipe.get("replicates", 1))
    return list(range(1, count + 1))


def fixed_test_fraction(recipe):
    return float(recipe.get("split", {}).get("fixed_test_fraction", 0.2))


def include_main_periodic(recipe):
    return str(recipe.get("include_main_periodic", DEFAULT_INCLUDE_MAIN_PERIODIC))


def include_test_periodic(recipe):
    return str(recipe.get("include_test_periodic", DEFAULT_INCLUDE_TEST_PERIODIC))


def use_visualization(recipe, override=None):
    if override is not None:
        return bool(override)
    return bool(recipe.get("visualization", False))


def sample_train_path(split_root, dataset_id_value, fraction, replicate):
    return (
        Path(split_root)
        / "sample_efficiency"
        / safe_filename(dataset_id_value)
        / replicate_label(replicate)
        / fraction_label(fraction)
        / "simulation.parquet"
    )


def source_fixed_test_path(split_root, dataset_id_value):
    return Path(split_root) / "sources" / safe_filename(dataset_id_value) / "fixed_test" / "simulation.parquet"


def balance_training_paths(split_root, pair_info, ratio, replicate):
    root = (
        Path(split_root)
        / "balance"
        / safe_filename(pair_info["solver"])
        / safe_filename(pair_info["pair_id"])
        / ratio_label(ratio)
        / replicate_label(replicate)
    )
    return {
        "left": root / safe_filename(pair_info["left_dataset_id"]) / "simulation.parquet",
        "right": root / safe_filename(pair_info["right_dataset_id"]) / "simulation.parquet",
    }


def balance_manifest_path(split_root, pair_info, ratio, replicate):
    return (
        Path(split_root)
        / "balance"
        / safe_filename(pair_info["solver"])
        / safe_filename(pair_info["pair_id"])
        / ratio_label(ratio)
        / replicate_label(replicate)
        / "split_manifest.json"
    )


def simplex_training_paths(split_root, solver_info, composition, replicate):
    root = (
        Path(split_root)
        / "simplex"
        / safe_filename(solver_info["solver"])
        / composition_label(composition)
        / replicate_label(replicate)
    )
    return {
        dataset_id_value: root / safe_filename(dataset_id_value) / "simulation.parquet"
        for dataset_id_value in solver_info["dataset_ids"]
    }


def simplex_manifest_path(split_root, solver_info, composition, replicate):
    return (
        Path(split_root)
        / "simplex"
        / safe_filename(solver_info["solver"])
        / composition_label(composition)
        / replicate_label(replicate)
        / "split_manifest.json"
    )


def sample_record_path(experiment_root, forecaster_id, dataset_id_value, fraction, replicate):
    return (
        Path(experiment_root)
        / DEFAULT_SAMPLE_ROOT
        / "records"
        / safe_filename(forecaster_id)
        / safe_filename(dataset_id_value)
        / fraction_label(fraction)
        / f"{replicate_label(replicate)}.json"
    )


def balance_record_path(experiment_root, forecaster_id, solver, pair_id, ratio, replicate):
    return (
        Path(experiment_root)
        / DEFAULT_BALANCE_ROOT
        / "records"
        / safe_filename(forecaster_id)
        / safe_filename(solver)
        / safe_filename(pair_id)
        / ratio_label(ratio)
        / f"{replicate_label(replicate)}.json"
    )


def simplex_record_path(experiment_root, forecaster_id, solver, composition, replicate):
    return (
        Path(experiment_root)
        / DEFAULT_SIMPLEX_ROOT
        / "records"
        / safe_filename(forecaster_id)
        / safe_filename(solver)
        / composition_label(composition)
        / f"{replicate_label(replicate)}.json"
    )


def fraction_label(value):
    percent = int(round(float(value) * 100))
    return f"f{percent:02d}"


def ratio_label(value):
    left = int(round(float(value) * 100))
    right = 100 - left
    return f"r{left:02d}_{right:02d}"


def composition_label(composition):
    percentages = [int(round(float(value) * 100)) for value in composition]
    return "c" + "_".join(f"{value:02d}" for value in percentages)


def ratio_column_label(ratio, pair_info):
    left = int(round(float(ratio) * 100))
    right = 100 - left
    return f"{left}/{right}"


def replicate_label(value):
    return f"rep_{int(value):02d}"


def clamp_fraction(value):
    value = float(value)
    if value <= 0 or value >= 1:
        raise ValueError("Fraction must be in the range (0, 1).")
    return value


def run_seed(recipe, *parts):
    seed = int(recipe.get("random_seed", 0))
    for part in parts:
        for char in str(part):
            seed = (seed * 131 + ord(char)) % (2**32 - 1)
    return int(seed)


def balance_seed(recipe, pair_info, ratio, replicate, side):
    return run_seed(recipe, pair_info["solver"], pair_info["pair_id"], ratio, replicate, side)


def simplex_seed(recipe, solver, composition, replicate, source_index):
    return run_seed(
        recipe,
        "simplex",
        solver,
        composition_label(composition),
        replicate,
        source_index,
    )


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def read_csv_allow_empty(path, columns):
    """Read an aggregate CSV, including legacy zero-byte partial-phase files."""
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(json_safe(payload), f, indent=2)


def add_cli_args(parser):
    parser.add_argument(
        "--root",
        default=None,
        help="Experiment root folder. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--recipe",
        default=None,
        help="Recipe JSON path. Defaults to ROOT/recipe.json.",
    )
    parser.add_argument("--validate-only", action="store_true", help="Validate recipe and source paths.")
    parser.add_argument(
        "--prepare-sample-splits-only",
        action="store_true",
        help="Prepare fixed-test and sample-efficiency train subset parquet files.",
    )
    parser.add_argument(
        "--prepare-balance-splits-only",
        action="store_true",
        help="Prepare mixed-balance train subset parquet files.",
    )
    parser.add_argument(
        "--prepare-simplex-splits-only",
        action="store_true",
        help="Prepare three-circuit simplex train subset parquet files.",
    )
    parser.add_argument("--run-sample", action="store_true", help="Run one sample-efficiency training cell.")
    parser.add_argument("--run-balance", action="store_true", help="Run one mixed-balance training cell.")
    parser.add_argument("--run-simplex", action="store_true", help="Run one three-circuit simplex cell.")
    parser.add_argument("--aggregate-only", action="store_true", help="Aggregate records and regenerate plots.")
    parser.add_argument(
        "--aggregate-simplex-only",
        action="store_true",
        help="Aggregate simplex records and regenerate triangular plots.",
    )
    parser.add_argument("--plot-only", action="store_true", help="Regenerate plots from aggregate CSV files.")
    parser.add_argument("--forecaster", default=None, help="Forecaster id for a single run.")
    parser.add_argument("--dataset", default=None, help="Dataset id for a sample-efficiency run.")
    parser.add_argument("--fraction", type=float, default=None, help="Training fraction for a sample run.")
    parser.add_argument("--replicate", type=int, default=None, help="One-based replicate number.")
    parser.add_argument("--solver", default=None, help="Solver id for a balance run.")
    parser.add_argument("--pair", default=None, help="Circuit-pair id for a balance run.")
    parser.add_argument("--ratio", type=float, default=None, help="Left-circuit training share for a balance run.")
    parser.add_argument(
        "--composition",
        default=None,
        help="Three comma-separated circuit fractions for a simplex run, such as 0.1,0.2,0.7.",
    )
    parser.add_argument(
        "--balance-total-fraction",
        type=float,
        default=None,
        help="Override auto plateau total fraction for balance split preparation/runs.",
    )
    parser.add_argument(
        "--visualization",
        dest="visualization",
        action="store_true",
        default=None,
        help="Force saving per-run prediction plots.",
    )
    parser.add_argument(
        "--no-visualization",
        dest="visualization",
        action="store_false",
        help="Disable per-run prediction plots.",
    )


def run_from_args(args):
    return run_training_data_experiment(
        root=args.root,
        recipe_path=args.recipe,
        validate_only=args.validate_only,
        prepare_sample_splits_only=args.prepare_sample_splits_only,
        prepare_balance_splits_only=args.prepare_balance_splits_only,
        prepare_simplex_splits_only=args.prepare_simplex_splits_only,
        run_sample=args.run_sample,
        run_balance=args.run_balance,
        run_simplex=args.run_simplex,
        aggregate_only=args.aggregate_only,
        aggregate_simplex_only=args.aggregate_simplex_only,
        plot_only=args.plot_only,
        forecaster=args.forecaster,
        dataset=args.dataset,
        fraction=args.fraction,
        replicate=args.replicate,
        solver=args.solver,
        pair=args.pair,
        ratio=args.ratio,
        composition=args.composition,
        balance_total_fraction=args.balance_total_fraction,
        visualization=args.visualization,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run AISAM forecaster training-data experiments.")
    add_cli_args(parser)
    args = parser.parse_args(argv)
    result = run_from_args(args)
    print(json.dumps(json_safe(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
