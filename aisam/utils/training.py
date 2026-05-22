"""Backward-compatible imports for AISAM training utilities.

The implementation is organized across:
- aisam.utils.simulation
- aisam.utils.forecaster_training
- aisam.utils.pipeline
"""

from aisam.utils.forecaster_training import (
    train_forecaster_from_simulation_config,
    train_forecaster_from_simulation,
    train_forecaster_random_stim_eval,
)
from aisam.utils.pipeline import load_experiment_recipe, run_recipe, training
from aisam.utils.simulation import (
    assign_temperatures,
    build_standard_stims,
    dated_assets_root,
    default_models_root,
    default_training_data_root,
    run_sanity_check_simulation,
    run_training_simulation,
    stimulation_cell_ranges,
)

__all__ = [
    "assign_temperatures",
    "build_standard_stims",
    "dated_assets_root",
    "default_models_root",
    "default_training_data_root",
    "load_experiment_recipe",
    "run_recipe",
    "run_sanity_check_simulation",
    "run_training_simulation",
    "stimulation_cell_ranges",
    "train_forecaster_from_simulation",
    "train_forecaster_from_simulation_config",
    "train_forecaster_random_stim_eval",
    "training",
]
