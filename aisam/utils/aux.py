import numpy as np
import json
from os import PathLike
from pathlib import Path

from aisam.model import defaults

CIRCUIT_PARAMETER_SCHEMAS = {
    "simple": ["delta"],
    "ccasr": ["alpha", "k", "n", "tau_delay", "h1", "h2", "c2", "delta"],
    "ccasr_noe": ["alpha", "k", "n", "tau_delay", "c2", "delta"],
    "ccasr_ode": ["alpha", "k", "n", "tau_delay", "c2", "delta"],
    "inverter": [
        "alpha",
        "beta",
        "k_tet",
        "k",
        "n",
        "n_tet",
        "tau_delay",
        "h1",
        "h2",
        "c2",
        "delta",
    ],
    "inverter_noe": [
        "alpha",
        "beta",
        "k_tet",
        "k",
        "n",
        "n_tet",
        "tau_delay",
        "c2",
        "delta",
    ],
    "ode_inverter": [
        "alpha",
        "beta",
        "k_tet",
        "k",
        "n",
        "n_tet",
        "tau_delay",
        "c2",
        "delta",
    ],
}

def repetitive_stim_maker(num_repeat,total_time,off_first = False):
    '''
    num_repeat: number of segments each made of repetitive cycles (e.g., if 2, [11110000])\\
    total_time: span of the stimulation experiment. For example, if 20, it means 20 cycles (and for a 5 min stim sampling, 100 mins).\\
    off_first: bool whether stim starts with off signal or not. Defaults is False.\\
    **Example**:\\
    num_repeat = 4 and total_time = 60 means [[15 ones and 15 zeros, 15 ones and 15 zeros]
    '''
    num_reps = int(np.floor(total_time/num_repeat))
    ons = np.repeat(1,num_reps)
    offs = np.repeat(0,num_reps)
    tile = np.hstack((offs,ons)) if off_first else np.hstack((ons,offs)) 
    stim_vec = np.tile(tile,int(num_repeat))
    stim_vec = stim_vec[:total_time]
    return stim_vec

def random_stim_maker(total_time,bias=[0.5,0.5]):
    '''
    A function to generate random sequences of stimulation. bias determines the weight of having (Green) stimulation {1} vs. red {0}
    total_time: total span of experiment in terms of number of stimulation (it would be 5*total_time in minutes)
    '''
    stim = np.random.choice(2,total_time,p=bias)
    return stim

def plot_w_bckgrnd(mega_res,stim_vec_tot,t_max,species = 'F',sampling=10,line_color= 'b',save=None,axes = False,ax_out=None):
    from aisam.utils.visualization_tools import plot_w_bckgrnd as _plot_w_bckgrnd

    return _plot_w_bckgrnd(
        mega_res,
        stim_vec_tot,
        t_max,
        species=species,
        sampling=sampling,
        line_color=line_color,
        save=save,
        axes=axes,
        ax_out=ax_out,
    )
        
        
def background_plotter(ax,stim_vec,sampling = 10, stim_period=5,averaged=False):
    from aisam.utils.visualization_tools import background_plotter as _background_plotter

    return _background_plotter(
        ax,
        stim_vec,
        sampling=sampling,
        stim_period=stim_period,
        averaged=averaged,
    )
            
def save_params(params,address):
    with open(address+'simulation_params.json', 'w') as f:
        json.dump(params, f)
        
def load_params(address):
    with open(address,'r') as f:
        params = json.load(f)
    return params

def config_generator(
    circuit,
    output_dir,
    t_max,
    sampling=10,
    interval_rate = 5,
    num_cells=1000,
    total_cell=None,
    noisy_total_cells=350,
    num_realizations=1,
    noisy_sims = 0,
    filename="config.json",
    params=None,
    save_params_file=True,
    **kwargs,
):
    """
    Generate a simulation/training config file for a supported circuit.

    Circuit parameters can be passed either through `params={...}` or directly
    as keyword arguments. Extra keyword arguments are preserved in the config as
    additional hyperparameters.

    Returns `(config_path, config)`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    circuit_key = defaults.normalize_circuit_name(circuit)
    required_params = CIRCUIT_PARAMETER_SCHEMAS.get(circuit_key)
    if required_params is None:
        raise ValueError(
            f"Unknown circuit {circuit!r}. Supported circuits: "
            f"{sorted(CIRCUIT_PARAMETER_SCHEMAS.keys())}"
        )

    circuit_params = dict(params or {})
    for key in required_params:
        if key in kwargs:
            circuit_params[key] = kwargs.pop(key)
    if circuit_key in {"ccasr_ode", "ode_inverter"}:
        for key in ("x0", "measurement_noise", "std"):
            if key in kwargs:
                circuit_params[key] = kwargs.pop(key)

    missing = [key for key in required_params if key not in circuit_params]
    if missing:
        raise ValueError(f"Missing required parameters for {circuit!r}: {missing}")

    if total_cell is not None:
        num_cells = total_cell
    num_cells = _validate_total_cells(num_cells)
    noisy_total_cells = _validate_total_cells(noisy_total_cells)

    circuit_params["t_max"] = t_max
    circuit_params["sampling"] = sampling

    params_path = None
    if save_params_file:
        params_path = output_dir / "simulation_params.json"
        with open(params_path, "w") as f:
            json.dump(_json_safe(circuit_params), f, indent=2)

    root_folder = str(output_dir)
    if not root_folder.endswith("/"):
        root_folder += "/"

    config = {
        "circuit": circuit_key,
        "label": circuit_key,
        "t_max": t_max,
        "sampling": sampling,
        "interval_rate":interval_rate,
        "num_cells": num_cells,
        "total_cell": num_cells,
        "total_cells": num_cells,
        "noisy_total_cells": noisy_total_cells,
        "num_realizations": num_realizations,
        "noisy_sims": noisy_sims,
        "root_folder": root_folder,
        "params": _json_safe(circuit_params),
        "circuit_parameters": _json_safe(circuit_params),
        
    }
    if params_path is not None:
        config["params_path"] = str(params_path)
    config.update(_json_safe(kwargs))

    config_path = output_dir / filename
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return config_path, config

def _validate_total_cells(total_cells):
    total_cells = int(total_cells)
    if total_cells < 200:
        raise ValueError("total_cell must be at least 200.")
    return total_cells

def sample_noisy_params(address, temperature=0.1, rng=None, clip_min=None):
    '''
    Load parameters from a JSON file and return a noisy copy where each parameter
    is sampled from a Normal distribution centered on its original value.

    address: path to the JSON parameter file.
    temperature: controls stochasticity; std is set to abs(value) * temperature.
    rng: optional numpy random generator for reproducibility.
    clip_min: optional lower-bound applied to sampled numeric values.
    '''
    if temperature < 0:
        raise ValueError("temperature must be non-negative.")

    params = load_params(address)
    generator = rng if rng is not None else np.random.default_rng()

    noisy_params = {}
    for key, value in params.items():
        if isinstance(value, (int, float)):
            std = abs(value) * temperature
            sampled = generator.normal(loc=value, scale=std)
            if clip_min is not None:
                sampled = max(sampled, clip_min)
            noisy_params[key] = float(sampled)
        else:
            noisy_params[key] = value

    return noisy_params

def sample_noisy_params_from_dict(params, temperature=0.1, rng=None, clip_min=None):
    '''
    Return a noisy copy of a parameter dictionary where each numeric parameter
    is sampled from a Normal distribution centered on its original value.

    params: parameter dictionary.
    temperature: controls stochasticity; std is set to abs(value) * temperature.
    rng: optional numpy random generator for reproducibility.
    clip_min: optional lower-bound applied to sampled numeric values.
    '''
    if temperature < 0:
        raise ValueError("temperature must be non-negative.")
    if not isinstance(params, dict):
        raise ValueError("params must be a dictionary.")

    generator = rng if rng is not None else np.random.default_rng()
    noisy_params = {}
    for key, value in params.items():
        if isinstance(value, (int, float)):
            std = abs(value) * temperature
            sampled = generator.normal(loc=value, scale=std)
            if clip_min is not None:
                sampled = max(sampled, clip_min)
            noisy_params[key] = float(sampled)
        else:
            noisy_params[key] = value

    return noisy_params

def save_stims(stims,address):
    with open(address + 'stims.json','w') as f:
        json.dump(stims,f)

def load_stims(address):
    with open(address,'r') as f:
        stims = json.load(f)
    return stims


def average_stochastic_trace(trajectory,sampling=10):
    len_trace = len(trajectory)
    averaged_trace = [np.mean(trajectory[i:i+sampling]) for i in np.arange(len_trace,step = sampling)]
    return averaged_trace

def sample_from_stochastic_trace(trajectory,sampling = 50):
    len_trace = len(trajectory)
    sampled_trace = [trajectory[i] for i in np.arange(len_trace,step=sampling)]
    return sampled_trace

def sample_trace_at_interval(trajectory, simulation_sampling, interval_minutes):
    """
    Sample a dense simulation trace at a user-given interval in minutes.

    `simulation_sampling` is the number of saved simulation values per minute.
    For example, `simulation_sampling=10` and `interval_minutes=5` samples
    indices 0, 50, 100, ...
    """
    if interval_minutes is None:
        return trajectory
    if simulation_sampling <= 0:
        raise ValueError("simulation_sampling must be positive.")
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive.")

    step = max(1, int(round(float(simulation_sampling) * float(interval_minutes))))
    return np.asarray(trajectory)[np.arange(0, len(trajectory), step)]

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
    return value
