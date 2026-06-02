import dill
import numpy as np
from os import PathLike
from pathlib import Path

from aisam.utils import aux


class RidgeRegressor:
    """Small multi-output ridge regressor with a scikit-learn-like interface."""

    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])

        penalty = self.alpha * np.eye(X.shape[1])
        if self.fit_intercept:
            penalty[0, 0] = 0.0

        self.coef_ = np.linalg.solve(X.T @ X + penalty, X.T @ y)
        return self

    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("Model has not been fit yet.")

        X = np.asarray(X, dtype=float)
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        return X @ self.coef_


class RegressionForecaster:
    """
    Windowed regression forecaster for simulated gene-expression trajectories.

    Inputs are flattened from:
    - past feature history: previous expression values for one or more species
    - past input history: previous stimulation values
    - future input plan: stimulation values over the prediction horizon

    Targets are the future expression values of one output species.
    """

    def __init__(
        self,
        past_feature_window,
        future_window=1,
        past_input_window=None,
        future_input_window=None,
        regressor=None,
        normalize=True,
        sampling=None,
        sample_interval_minutes=None,
    ):
        self.past_feature_window = int(past_feature_window)
        self.future_window = int(future_window)
        self.past_input_window = int(past_input_window or past_feature_window)
        self.future_input_window = int(future_input_window or future_window)
        self.regressor = regressor if regressor is not None else RidgeRegressor()
        self.normalize = normalize
        self.sampling = sampling
        self.sample_interval_minutes = sample_interval_minutes
        self.x_mean_ = None
        self.x_std_ = None
        self.y_mean_ = None
        self.y_std_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if self.normalize:
            X, y = self._fit_normalizers(X, y)

        self.regressor.fit(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.normalize:
            X = self._normalize_x(X)

        pred = self.regressor.predict(X)
        if self.normalize:
            pred = self._denormalize_y(pred)
        return pred

    def evaluate(self, X, y):
        pred = self.predict(X)
        return regression_metrics(y, pred)

    def _fit_normalizers(self, X, y):
        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0)
        self.x_std_[self.x_std_ == 0] = 1.0

        self.y_mean_ = y.mean(axis=0)
        self.y_std_ = y.std(axis=0)
        self.y_std_[self.y_std_ == 0] = 1.0

        return self._normalize_x(X), (y - self.y_mean_) / self.y_std_

    def _normalize_x(self, X):
        return (X - self.x_mean_) / self.x_std_

    def _denormalize_y(self, y):
        return y * self.y_std_ + self.y_mean_


def load_cells(path):
    """Load a saved `experiment.Cells` pickle produced by `experiment.save_cells`."""
    with open(path, "rb") as f:
        return dill.load(f)


def load_simulation_dataframe(path):
    """Load a long-form simulation parquet file."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Reading simulation parquet files requires pandas and pyarrow.") from exc

    return pd.read_parquet(path)


def cells_to_sequences(
    cells,
    feature_species=None,
    output_species="F",
    input_dim=1,
):
    """
    Convert saved Cell_sim objects into per-realization time-series dictionaries.

    Returns a list of dicts with:
    - features: array with shape (time, num_features)
    - inputs: array with shape (time, input_dim)
    - output: array with shape (time,)
    - cell_id, run_index, realization_index metadata
    """
    sequences = []

    for cell_id in _sorted_cell_ids(cells):
        cell = cells[cell_id]
        species_names = list(cell.expressions.keys())
        selected_species = _normalize_species_list(feature_species, species_names)

        if output_species not in cell.expressions:
            raise ValueError(f"Output species {output_species!r} not found in cell {cell_id}.")
        missing = [name for name in selected_species if name not in cell.expressions]
        if missing:
            raise ValueError(f"Feature species missing in cell {cell_id}: {missing}")

        num_runs = len(cell.stims)
        for run_index in range(num_runs):
            output_runs = _as_realization_matrix(cell.expressions[output_species][run_index])
            num_realizations = output_runs.shape[0]

            for realization_index in range(num_realizations):
                feature_cols = []
                for species in selected_species:
                    species_runs = _as_realization_matrix(cell.expressions[species][run_index])
                    if species_runs.shape[0] == 1 and num_realizations > 1:
                        values = species_runs[0]
                    else:
                        values = species_runs[realization_index]
                    feature_cols.append(values)

                features = np.column_stack(feature_cols).astype(float)
                output = output_runs[realization_index].astype(float)
                inputs = stim_to_input_trace(cell.stims[run_index], len(output), input_dim=input_dim)

                sequences.append(
                    {
                        "features": features,
                        "inputs": inputs,
                        "output": output,
                        "cell_id": cell_id,
                        "run_index": run_index,
                        "realization_index": realization_index,
                    }
                )

    return sequences


def dataframe_to_sequences(
    dataframe,
    feature_species=None,
    output_species="F",
    input_dim=1,
    value_column="value",
):
    """
    Convert a long-form simulation dataframe into per-realization sequences.

    Expected columns are:
    cell_id, realization, species, stim, time, value
    """
    required = {"cell_id", "realization", "species", "stim", "time", value_column}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Simulation dataframe is missing required columns: {sorted(missing)}")

    sequences = []
    species_names = list(dataframe["species"].drop_duplicates())
    selected_species = _normalize_species_list(feature_species, species_names)
    missing_species = [name for name in selected_species if name not in species_names]
    if output_species not in species_names:
        missing_species.append(output_species)
    if missing_species:
        raise ValueError(f"Species missing from simulation dataframe: {sorted(set(missing_species))}")

    grouped = dataframe.groupby(["cell_id", "realization"], sort=False)
    for (cell_id, realization), group in grouped:
        output_group = _species_frame(group, output_species, value_column)
        reference_times = output_group["time"].to_numpy()
        output = output_group[value_column].to_numpy(dtype=float)
        stim = output_group["stim"].to_numpy(dtype=float).reshape(-1, input_dim)

        feature_cols = []
        for species in selected_species:
            species_group = _species_frame(group, species, value_column)
            species_times = species_group["time"].to_numpy()
            if not np.array_equal(reference_times, species_times):
                raise ValueError(
                    f"Species {species!r} in cell {cell_id!r}, realization {realization!r} "
                    "does not share the same time grid as the output species."
                )
            feature_cols.append(species_group[value_column].to_numpy(dtype=float))

        sequences.append(
            {
                "features": np.column_stack(feature_cols).astype(float),
                "inputs": stim,
                "output": output,
                "cell_id": cell_id,
                "run_index": 0,
                "realization_index": _realization_index(realization),
            }
        )

    return sequences


def make_window_dataset(
    sequences,
    past_feature_window,
    future_window=1,
    past_input_window=None,
    future_input_window=None,
    stride=1,
):
    """Build a supervised regression matrix from trajectory sequences."""
    past_feature_window = int(past_feature_window)
    future_window = int(future_window)
    past_input_window = int(past_input_window or past_feature_window)
    future_input_window = int(future_input_window or future_window)
    stride = int(stride)

    X_rows = []
    y_rows = []
    meta = []

    history_window = max(past_feature_window, past_input_window)
    for seq_index, seq in enumerate(sequences):
        features = np.asarray(seq["features"], dtype=float)
        inputs = np.asarray(seq["inputs"], dtype=float)
        output = np.asarray(seq["output"], dtype=float)

        _validate_sequence_shapes(features, inputs, output)
        stop = len(output) - max(future_window, future_input_window) + 1

        for t in range(history_window, stop, stride):
            feature_hist = features[t - past_feature_window:t].reshape(-1)
            input_hist = inputs[t - past_input_window:t].reshape(-1)
            future_inputs = inputs[t:t + future_input_window].reshape(-1)
            target = output[t:t + future_window].reshape(-1)

            X_rows.append(np.concatenate([feature_hist, input_hist, future_inputs]))
            y_rows.append(target)
            meta.append(
                {
                    "sequence_index": seq_index,
                    "time_index": t,
                    "cell_id": seq.get("cell_id"),
                    "run_index": seq.get("run_index"),
                    "realization_index": seq.get("realization_index"),
                }
            )

    if not X_rows:
        raise ValueError("No training windows were generated. Check window sizes and trace length.")

    return np.vstack(X_rows), np.vstack(y_rows), meta


def sample_sequences_at_interval(sequences, sampling, sample_interval_minutes=None):
    """
    Downsample feature, input, and output traces before windowing.

    `sampling` is the dense simulation sampling rate in values per minute.
    For `sampling=10` and `sample_interval_minutes=5`, indices 0, 50, 100, ...
    are kept for every sequence array.
    """
    if sample_interval_minutes is None:
        return list(sequences)
    if sampling is None:
        raise ValueError("sampling must be provided when sample_interval_minutes is set.")

    sampled = []
    for seq in sequences:
        sampled_seq = dict(seq)
        sampled_seq["features"] = aux.sample_trace_at_interval(
            seq["features"],
            simulation_sampling=sampling,
            interval_minutes=sample_interval_minutes,
        )
        sampled_seq["inputs"] = aux.sample_trace_at_interval(
            seq["inputs"],
            simulation_sampling=sampling,
            interval_minutes=sample_interval_minutes,
        )
        sampled_seq["output"] = aux.sample_trace_at_interval(
            seq["output"],
            simulation_sampling=sampling,
            interval_minutes=sample_interval_minutes,
        )
        sampled_seq["sample_interval_minutes"] = float(sample_interval_minutes)
        sampled_seq["source_sampling"] = int(sampling)
        sampled.append(sampled_seq)
    return sampled


def train_regression_forecaster(
    train_data,
    past_feature_window,
    future_window=1,
    past_input_window=None,
    future_input_window=None,
    feature_species=None,
    output_species="F",
    regressor=None,
    normalize=True,
    stride=1,
    validation_fraction=0.2,
    random_state=None,
    sampling=None,
    sample_interval_minutes=None,
):
    """
    Train a RegressionForecaster from saved cells, a pickle path, or prebuilt sequences.

    Cells are shuffled and split before windowing, so windows from one cell cannot
    appear in both the training and validation sets.

    Returns `(forecaster, metrics, dataset)`. Metrics contains train metrics and,
    when validation_fraction > 0, validation metrics. Dataset contains the windowed
    train/validation arrays plus the split sequences for downstream inspection.
    """
    sequences = _coerce_sequences(train_data, feature_species, output_species)
    sequences = sample_sequences_at_interval(
        sequences,
        sampling=sampling,
        sample_interval_minutes=sample_interval_minutes,
    )
    train_sequences, validation_sequences = split_sequences_by_cell(
        sequences,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )

    X_train, y_train, train_meta = make_window_dataset(
        train_sequences,
        past_feature_window=past_feature_window,
        future_window=future_window,
        past_input_window=past_input_window,
        future_input_window=future_input_window,
        stride=stride,
    )

    forecaster = RegressionForecaster(
        past_feature_window=past_feature_window,
        future_window=future_window,
        past_input_window=past_input_window,
        future_input_window=future_input_window,
        regressor=regressor,
        normalize=normalize,
        sampling=sampling,
        sample_interval_minutes=sample_interval_minutes,
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
        X_val, y_val, val_meta = make_window_dataset(
            validation_sequences,
            past_feature_window=past_feature_window,
            future_window=future_window,
            past_input_window=past_input_window,
            future_input_window=future_input_window,
            stride=stride,
        )
        val_predictions = forecaster.predict(X_val)
        metrics["validation"] = regression_metrics(y_val, val_predictions)
        dataset.update(
            {
                "X_validation": X_val,
                "y_validation": y_val,
                "validation_meta": val_meta,
                "validation_predictions": val_predictions,
            }
        )

    return forecaster, metrics, dataset


def evaluate_regression_forecaster(
    forecaster,
    eval_data,
    feature_species=None,
    output_species="F",
    stride=1,
    sampling=None,
    sample_interval_minutes=None,
):
    """
    Evaluate a trained RegressionForecaster on saved cells, a pickle path, or sequences.

    Returns `(metrics, predictions, dataset)`.
    """
    sequences = _coerce_sequences(eval_data, feature_species, output_species)
    if sampling is None:
        sampling = forecaster.sampling
    if sample_interval_minutes is None:
        sample_interval_minutes = forecaster.sample_interval_minutes
    sequences = sample_sequences_at_interval(
        sequences,
        sampling=sampling,
        sample_interval_minutes=sample_interval_minutes,
    )
    X, y, meta = make_window_dataset(
        sequences,
        past_feature_window=forecaster.past_feature_window,
        future_window=forecaster.future_window,
        past_input_window=forecaster.past_input_window,
        future_input_window=forecaster.future_input_window,
        stride=stride,
    )
    predictions = forecaster.predict(X)
    metrics = regression_metrics(y, predictions)
    dataset = {"X": X, "y": y, "meta": meta, "sequences": sequences}
    return metrics, predictions, dataset


def regression_metrics(y_true, y_pred):
    """Return common multi-output regression metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred

    mse = float(np.mean(residual ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residual)))

    denom = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
    r2 = np.nan if denom == 0 else float(1 - np.sum(residual ** 2) / denom)

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def split_sequences_by_cell(sequences, validation_fraction=0.2, random_state=None):
    """
    Shuffle and split sequences by cell id before windowing.

    Every sequence with the same `cell_id` stays in the same split. For custom
    prebuilt sequences without a `cell_id`, each sequence is treated as its own
    independent group.
    """
    sequences = list(sequences)
    if not sequences:
        raise ValueError("No sequences were provided.")

    validation_fraction = float(validation_fraction)
    if validation_fraction < 0 or validation_fraction >= 1:
        raise ValueError("validation_fraction must be in the range [0, 1).")
    if validation_fraction == 0:
        return sequences, []

    groups = {}
    for seq_index, seq in enumerate(sequences):
        group_id = seq.get("cell_id", seq_index)
        groups.setdefault(group_id, []).append(seq)

    group_ids = np.array(list(groups.keys()), dtype=object)
    rng = np.random.default_rng(random_state)
    rng.shuffle(group_ids)

    num_validation = int(np.ceil(len(group_ids) * validation_fraction))
    if len(group_ids) > 1:
        num_validation = min(max(num_validation, 1), len(group_ids) - 1)
    else:
        num_validation = 0

    validation_ids = set(group_ids[:num_validation])
    train_sequences = []
    validation_sequences = []
    for group_id in group_ids:
        if group_id in validation_ids:
            validation_sequences.extend(groups[group_id])
        else:
            train_sequences.extend(groups[group_id])

    return train_sequences, validation_sequences


def stim_to_input_trace(stim_vec, trace_length, input_dim=1):
    """Expand a lower-frequency stimulation vector to match expression trace length."""
    stim = np.asarray(stim_vec, dtype=float)
    if stim.ndim == 1:
        stim = stim.reshape(-1, 1)
    if stim.shape[1] != input_dim:
        if input_dim == 1:
            stim = stim[:, :1]
        else:
            raise ValueError(f"Expected input_dim={input_dim}, got stimulation shape {stim.shape}.")

    repeats = int(np.ceil(trace_length / len(stim)))
    expanded = np.repeat(stim, repeats, axis=0)
    return expanded[:trace_length]


def _coerce_sequences(data, feature_species, output_species):
    if isinstance(data, (str, PathLike)):
        path = Path(data)
        if path.suffix == ".parquet":
            return dataframe_to_sequences(
                load_simulation_dataframe(path),
                feature_species=feature_species,
                output_species=output_species,
            )
        return cells_to_sequences(load_cells(path), feature_species, output_species)
    if isinstance(data, dict) and data and hasattr(next(iter(data.values())), "expressions"):
        return cells_to_sequences(data, feature_species, output_species)
    if hasattr(data, "columns") and {"cell_id", "realization", "species", "stim", "time"}.issubset(data.columns):
        return dataframe_to_sequences(data, feature_species, output_species)
    return data


def _species_frame(group, species, value_column):
    frame = group[group["species"] == species].sort_values("time")
    if frame.empty:
        raise ValueError(f"Species {species!r} is missing from a cell/realization group.")
    if frame["time"].duplicated().any():
        raise ValueError(f"Species {species!r} has duplicate time points in a cell/realization group.")
    return frame


def _normalize_species_list(feature_species, species_names):
    if feature_species is None:
        return list(species_names)
    if isinstance(feature_species, str):
        return [feature_species]
    return list(feature_species)


def _realization_index(realization):
    try:
        return int(realization) - 1
    except (TypeError, ValueError):
        return realization


def _as_realization_matrix(values):
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim == 2:
        return array
    raise ValueError(f"Expected a 1D or 2D expression trace, got shape {array.shape}.")


def _sorted_cell_ids(cells):
    def key(cell_id):
        try:
            return int(cell_id)
        except (TypeError, ValueError):
            return cell_id

    return sorted(cells.keys(), key=key)


def _validate_sequence_shapes(features, inputs, output):
    if features.ndim != 2:
        raise ValueError("features must have shape (time, num_features).")
    if inputs.ndim != 2:
        raise ValueError("inputs must have shape (time, input_dim).")
    if output.ndim != 1:
        raise ValueError("output must have shape (time,).")
    if not (len(features) == len(inputs) == len(output)):
        raise ValueError("features, inputs, and output must have matching time lengths.")
