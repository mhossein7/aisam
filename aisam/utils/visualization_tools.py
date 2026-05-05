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
            ax.axvspan(x_start, x_end, facecolor=color, alpha=0.2)

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
        ax.axvspan(x_start, x_end, facecolor=color, alpha=0.2)


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
        x_future,
        prediction[:future_window],
        color="tab:blue",
        linewidth=1.8,
        linestyle=":",
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
    errors = np.mean((y_true - y_pred) ** 2, axis=1)

    rng = np.random.default_rng(random_state)
    indices = {
        "random": int(rng.integers(len(errors))),
        "best": int(np.argmin(errors)),
        "worst": int(np.argmax(errors)),
    }

    paths = {}
    for name, index in indices.items():
        meta = dataset["validation_meta"][index]
        sequence = dataset["validation_sequences"][meta["sequence_index"]]
        save_path = output_dir / f"{name}_evaluation.svg"
        plot_forecaster_window(
            sequence=sequence,
            time_index=meta["time_index"],
            prediction=y_pred[index],
            ground_truth=y_true[index],
            forecaster=forecaster,
            save_path=save_path,
            title=f"{name.title()} validation example | MSE={errors[index]:.4g}",
            output_species=output_species,
        )
        paths[name] = save_path

    return paths


def _plot_input_background(ax, inputs, x_values, minute_step):
    if len(inputs) == 0:
        return
    half_step = minute_step / 2
    for x, val in zip(x_values, inputs):
        color = "green" if val >= 0.5 else "red"
        ax.axvspan(x - half_step, x + half_step, facecolor=color, alpha=0.16, linewidth=0)
