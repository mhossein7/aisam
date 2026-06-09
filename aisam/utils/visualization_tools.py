import numpy as np
from matplotlib import pyplot as plt


def plot_w_bckgrnd(
    mega_res,
    stim_vec_tot,
    t_max,
    species="F",
    sampling=10,
    line_color="b",
    save=None,
    axes=False,
    ax_out=None,
):
    """
    Plot simulated expression traces with red/green stimulation background.
    """
    if ax_out is not None:
        ax = ax_out
        fig = ax.figure
    else:
        fig, ax = plt.subplots(figsize=(8, 4))

    for j in range(len(stim_vec_tot)):
        for i, val in enumerate(stim_vec_tot[j]):
            x_start = j * t_max * sampling + i * 5 * sampling
            x_end = x_start + 5 * sampling
            color = "green" if val == 1 else "red"
            _input_span(ax, x_start, x_end, color=color, alpha=0.2)

    for k in range(len(mega_res.keys())):
        results = mega_res[f"cell {k+1}"]
        for i in range(len(results)):
            ax.plot(
                np.arange(t_max * i * sampling, t_max * (i + 1) * sampling),
                results[i][species],
                color=line_color,
                linewidth=0.5,
            )

    if save is not None:
        fig.savefig(save["path"], dpi=300, format="svg")
    elif axes:
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("GFP (molecule count)")
        return ax
    else:
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("GFP (molecule count)")
        plt.tight_layout()
        plt.show()


def background_plotter(ax, stim_vec, sampling=10, stim_period=5, averaged=False):
    if averaged:
        sampling = 1
    for i, val in enumerate(stim_vec):
        x_start = sampling * i * stim_period
        x_end = x_start + sampling * stim_period
        color = "green" if val == 1 else "red"
        _input_span(ax, x_start, x_end, color=color, alpha=0.2)


def plot_forecaster_window(
    sequence,
    time_index,
    prediction,
    ground_truth,
    forecaster,
    save_path,
    title=None,
    output_species="F",
):
    """
    Save a single forecaster evaluation window as an SVG.

    The plot shows stimulation background, past output expression, future ground
    truth, and future model prediction.
    """
    past_window = forecaster.past_feature_window
    future_window = forecaster.future_window
    future_input_window = forecaster.future_input_window
    start = time_index - past_window
    stop = time_index + max(future_window, future_input_window)
    if start < 0:
        raise ValueError("time_index is too early for the forecaster history window.")

    output = np.asarray(sequence["output"], dtype=float)
    inputs = np.asarray(sequence["inputs"], dtype=float).reshape(len(output), -1)[:, 0]
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    ground_truth = np.asarray(ground_truth, dtype=float).reshape(-1)

    minute_step = forecaster.sample_interval_minutes or 1
    x_window = np.arange(start, stop) * minute_step
    x_future = np.arange(time_index, time_index + future_window) * minute_step
    x_truth = np.arange(start, time_index + future_window) * minute_step
    truth = np.concatenate([output[start:time_index], ground_truth[:future_window]])
    x_prediction = x_future
    prediction_trace = prediction[:future_window]
    if time_index > start:
        x_prediction = np.concatenate(([(time_index - 1) * minute_step], x_future))
        prediction_trace = np.concatenate(([output[time_index - 1]], prediction_trace))

    fig, ax = plt.subplots(figsize=(9, 4))
    _plot_input_background(ax, inputs[start:stop], x_window, minute_step)

    ax.plot(
        x_truth,
        truth,
        color="black",
        linewidth=1.5,
        label=f"{output_species} ground truth",
    )
    ax.plot(
        x_prediction,
        prediction_trace,
        color="tab:blue",
        linewidth=1.8,
        linestyle=":",
        marker="o" if future_window == 1 else None,
        markersize=3,
        label="prediction",
    )
    ax.axvline(time_index * minute_step, color="0.35", linewidth=1, linestyle="--")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(output_species)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, format="svg")
    plt.close(fig)


def plot_forecaster_evaluation_examples(
    forecaster,
    dataset,
    output_dir,
    output_species="F",
    random_state=None,
):
    """
    Save random, best, and worst validation examples for a trained forecaster.
    """
    from pathlib import Path

    if "X_validation" not in dataset or "validation_predictions" not in dataset:
        raise ValueError("Visualization requires a validation dataset with predictions.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(dataset["y_validation"], dtype=float)
    y_pred = np.asarray(dataset["validation_predictions"], dtype=float)
    finite_mask = np.isfinite(y_true).all(axis=1) & np.isfinite(y_pred).all(axis=1)
    if not finite_mask.any():
        return {}
    finite_indices = np.flatnonzero(finite_mask)
    finite_rmse = np.sqrt(np.mean((y_true[finite_mask] - y_pred[finite_mask]) ** 2, axis=1))

    rng = np.random.default_rng(random_state)
    indices = {
        "random": int(finite_indices[int(rng.integers(len(finite_indices)))]),
        "best": int(finite_indices[int(np.argmin(finite_rmse))]),
        "worst": int(finite_indices[int(np.argmax(finite_rmse))]),
    }
    per_window_rmse = np.full(y_true.shape[0], np.nan, dtype=float)
    per_window_rmse[finite_mask] = finite_rmse

    paths = {}
    for name, index in indices.items():
        meta = dataset["validation_meta"][index]
        sequence = dataset["validation_sequences"][meta["sequence_index"]]
        save_path = output_dir / f"{name}_prediction.svg"
        plot_forecaster_window(
            sequence=sequence,
            time_index=meta["time_index"],
            prediction=y_pred[index],
            ground_truth=y_true[index],
            forecaster=forecaster,
            save_path=save_path,
            title=f"{name.title()} prediction | RMSE={per_window_rmse[index]:.4g}",
            output_species=output_species,
        )
        paths[f"{name}_prediction"] = save_path

    return paths


def plot_error_distribution(errors, save_path, title="Error distribution"):
    """
    Save a histogram of log10 per-window RMSE values.
    """
    from pathlib import Path

    errors = np.asarray(errors, dtype=float).reshape(-1)
    errors = errors[np.isfinite(errors)]
    if len(errors) == 0:
        errors = np.asarray([0.0], dtype=float)
    log_errors = np.log10(np.clip(errors, np.finfo(float).tiny, None))
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = min(40, max(10, int(np.sqrt(len(log_errors)))))
    ax.hist(log_errors, bins=bins, color="0.25", edgecolor="0.25", alpha=0.82)
    ax.axvline(float(np.mean(log_errors)), color="tab:blue", linewidth=1.5, linestyle="--", label="mean")
    ax.set_xlabel("log10(RMSE)")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, format="svg")
    plt.close(fig)
    return save_path


def plot_sanity_simulation_group(
    cells,
    cell_ids,
    species,
    save_path,
    title,
    sampling=10,
):
    """
    Save a sanity-check panel with cells in rows and species in columns.
    """
    from pathlib import Path

    species = [species] if isinstance(species, str) else list(species)
    fig, axes = plt.subplots(
        nrows=len(cell_ids),
        ncols=len(species),
        figsize=(4.2 * len(species), 1.45 * len(cell_ids)),
        squeeze=False,
        sharex=True,
    )
    for row, cell_id in enumerate(cell_ids):
        cell = cells[str(cell_id)]
        stim_vec = np.asarray(cell.stims[0], dtype=float)
        for col, species_name in enumerate(species):
            ax = axes[row][col]
            traces = _traces_for_plot(cell.expressions[species_name][0])
            x = np.arange(traces.shape[1]) / float(sampling)
            _plot_stim_background_for_trace(ax, stim_vec, traces.shape[1], sampling)
            for trace in traces:
                ax.plot(x, trace, color="black", linewidth=1)
            if row == 0:
                ax.set_title(species_name)
            if col == 0:
                ax.set_ylabel(f"cell {cell_id}")
            if row == len(cell_ids) - 1:
                ax.set_xlabel("Time (min)")

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, format="svg")
    plt.close(fig)
    return save_path


def _plot_input_background(ax, inputs, x_values, minute_step):
    if len(inputs) == 0:
        return
    half_step = minute_step / 2
    for x, val in zip(x_values, inputs):
        color = "green" if val >= 0.5 else "red"
        _input_span(ax, x - half_step, x + half_step, color=color, alpha=0.16)


def _plot_stim_background_for_trace(ax, stim_vec, trace_length, sampling):
    if len(stim_vec) == 0:
        return
    minutes_per_stim = trace_length / float(sampling * len(stim_vec))
    for i, val in enumerate(stim_vec):
        color = "green" if val >= 0.5 else "red"
        _input_span(ax, i * minutes_per_stim, (i + 1) * minutes_per_stim, color=color, alpha=0.16)


def _traces_for_plot(trace):
    trace = np.asarray(trace, dtype=float)
    if trace.ndim == 1:
        return trace.reshape(1, -1)
    if trace.ndim == 2:
        if trace.shape[0] <= 5:
            return trace
        return trace.mean(axis=0).reshape(1, -1)
    raise ValueError(f"Expected 1D or 2D trace, got shape {trace.shape}.")


def _input_span(ax, x_start, x_end, color, alpha):
    ax.axvspan(
        x_start,
        x_end,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=0,
    )
