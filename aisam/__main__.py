import argparse
import json
from os import PathLike

import numpy as np

from aisam.utils.training import run_recipe


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

    args = parser.parse_args(argv)
    if args.command == "simulate":
        result = run_recipe(root_folder=args.path, progress=not args.no_progress)
        print(json.dumps(_json_safe(result), indent=2))
        return 0

    parser.print_help()
    return 0


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
