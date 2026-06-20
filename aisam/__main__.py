import argparse
import json
from os import PathLike

import numpy as np

from aisam.experiments.forecaster_comparison import (
    add_cli_args as _add_forecaster_comparison_args,
    run_from_args as _run_forecaster_comparison_from_args,
)
from aisam.utils.forecaster_training import cross_test_forecaster, train_forecaster
from aisam.utils.pipeline import run_recipe


def main(argv=None):
    parser = argparse.ArgumentParser(prog="aisam")
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="Run a recipe-driven AISAM experiment.")
    _add_run_args(run)
    simulate = subparsers.add_parser("simulate", help="Compatibility alias for `aisam run`.")
    _add_run_args(simulate)
    train = subparsers.add_parser(
        "train",
        help="Compatibility command for policy-driven training from existing simulation data.",
    )
    _add_train_args(train)
    predict = subparsers.add_parser(
        "predict",
        help="Train/load a forecaster and evaluate it on an external simulation dataset.",
    )
    _add_predict_args(predict)
    compare = subparsers.add_parser(
        "compare-forecasters",
        help="Run a matrix-style forecaster comparison experiment.",
    )
    _add_forecaster_comparison_args(compare)
    comparison_alias = subparsers.add_parser(
        "forecaster-comparison",
        help="Compatibility alias for `aisam compare-forecasters`.",
    )
    _add_forecaster_comparison_args(comparison_alias)

    args = parser.parse_args(argv)
    if args.command in {"run", "simulate"}:
        result = run_recipe(root_folder=args.path, progress=not args.no_progress)
        print(json.dumps(_json_safe(result), indent=2))
        return 0
    if args.command == "train":
        result = train_forecaster(
            path=args.path,
            config=args.config,
            output_root=args.output_root,
            visualization=args.visualization,
            random_state=args.random_state,
            include_noisy=args.include_noisy,
            include_noisy_periodic=args.include_noisy_periodic,
            include_main_periodic=args.include_main_periodic,
            label=args.label,
        )
        print(json.dumps(_json_safe(result), indent=2))
        return 0
    if args.command == "predict":
        result = cross_test_forecaster(
            model=args.model,
            training_data=args.training_data,
            test_data=args.test_data,
            trained_model=args.trained_model,
            output_root=args.output_root,
            visualization=args.visualization,
            random_state=args.random_state,
            include_main_periodic=args.include_main_periodic,
            label=args.label,
        )
        print(json.dumps(_json_safe(result), indent=2))
        return 0
    if args.command in {"compare-forecasters", "forecaster-comparison"}:
        result = _run_forecaster_comparison_from_args(args)
        print(json.dumps(_json_safe(result), indent=2))
        return 0

    parser.print_help()
    return 0


def _add_run_args(parser):
    parser.add_argument(
        "--path",
        default=None,
        help="Folder containing recipe.json. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable per-cell progress bars.",
    )


def _add_train_args(parser):
    parser.add_argument(
        "--path",
        default=None,
        help="Simulation parquet/pickle path or a run folder. Defaults to cwd.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Forecaster model_config.json path or model type.",
    )
    parser.add_argument(
        "--include-noisy",
        default="none",
        choices=("train", "eval", "none"),
        help="Use noisy random-stim simulations for training, evaluation, or ignore them.",
    )
    parser.add_argument(
        "--include-noisy-periodic",
        default="none",
        choices=("train", "eval", "none"),
        help="Use noisy periodic/repetitive simulations for training, evaluation, or ignore them.",
    )
    parser.add_argument(
        "--include-main-periodic",
        default="eval",
        choices=("train", "eval", "none"),
        help="Use main periodic/repetitive simulations for training, evaluation, or ignore them.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional folder where the model run folder should be created.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional model run label prefix.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Optional random seed for train/evaluation split and example selection.",
    )
    parser.add_argument(
        "--visualization",
        action="store_true",
        help="Save representative examples and error distribution plots.",
    )


def _add_predict_args(parser):
    parser.add_argument(
        "--model",
        default=None,
        help="Forecaster model type or model config JSON path, e.g. regressor or lstm_encoder_decoder.",
    )
    parser.add_argument(
        "--training-data",
        default=None,
        help="Training simulation parquet/pickle path or run folder.",
    )
    parser.add_argument(
        "--test-data",
        required=True,
        help="External test simulation parquet/pickle path or run folder.",
    )
    parser.add_argument(
        "--trained-model",
        default=None,
        help="Path to an existing trained model.pkl. Skips model training.",
    )
    parser.add_argument(
        "--include-main-periodic",
        default="eval",
        choices=("train", "eval", "none"),
        help="Use training-data periodic/repetitive cells for training, evaluation, or ignore them.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional folder where cross-testing outputs should be saved.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional model label used in the cross-testing output path.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Optional random seed for train/holdout split and example selection.",
    )
    parser.add_argument(
        "--visualization",
        action="store_true",
        default=None,
        help="Save training-holdout and test plots. Enabled by default for predict.",
    )


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


if __name__ == "__main__":
    raise SystemExit(main())
