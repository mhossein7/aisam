import argparse
import json
from os import PathLike

import numpy as np

from aisam.utils.forecaster_training import train_forecaster_from_simulation_config
from aisam.utils.pipeline import run_recipe


def main(argv=None):
    parser = argparse.ArgumentParser(prog="aisam")
    subparsers = parser.add_subparsers(dest="command")

    simulate = subparsers.add_parser("simulate", help="Run an experiment from recipe.json.")
    simulate.add_argument(
        "--path",
        default=None,
        help="Folder containing recipe.json. Defaults to the current working directory.",
    )
    simulate.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable per-cell progress bars.",
    )
    train = subparsers.add_parser("train", help="Train a forecaster from existing simulation data.")
    _add_train_args(train)

    args = parser.parse_args(argv)
    if args.command == "simulate":
        result = run_recipe(root_folder=args.path, progress=not args.no_progress)
        print(json.dumps(_json_safe(result), indent=2))
        return 0
    if args.command == "train":
        result = train_forecaster_from_simulation_config(
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

    parser.print_help()
    return 0


def _add_train_args(parser):
    parser.add_argument(
        "--path",
        default=None,
        help="Main simulation.pkl path or a folder containing simulation.pkl. Defaults to cwd.",
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
