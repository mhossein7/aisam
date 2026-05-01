import argparse
import json
from datetime import datetime
from pathlib import Path
from aisam.utils import aux
from aisam.model import sims

def build_standard_stims(total_time):
    stims = {}
    for i in range(1, 901):
        stims[f"cell {i}"] = aux.random_stim_maker(total_time).tolist()
    for i in range(901, 951):
        stims[f"cell {i}"] = aux.repetitive_stim_maker(num_repeat=4, total_time=total_time, off_first=True).tolist()
    for i in range(951, 1001):
        stims[f"cell {i}"] = aux.repetitive_stim_maker(num_repeat=4, total_time=total_time, off_first=False).tolist()
    return stims

def assign_temperatures(num_sims, temperatures):
    if num_sims <= 0:
        return []
    if not temperatures:
        return [0.1] * num_sims
    base = num_sims // len(temperatures)
    rem = num_sims % len(temperatures)
    assigned = []
    for i, temp in enumerate(temperatures):
        count = base + (1 if i < rem else 0)
        assigned.extend([float(temp)] * count)
    return assigned


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gene expression simulation")
    
    parser.add_argument("--label", type=str, default="CcaSR",
                        help="Name of the simulation")
    parser.add_argument("--config", type=str, default="",
                        help="Address to the config.json file")
    parser.add_argument("--standard-pipeline", action="store_true",
                        help="Run the standard 1000-cell stimulation pipeline and save outputs in assets/Simulations.")
    parser.add_argument("--noisy-sims", type=int, default=0,
                        help="Number of additional noisy simulations to run after standard pipeline.")
    parser.add_argument("--temperatures", type=float, nargs="*", default=None,
                        help="Temperature vector used to divide noisy simulations across stochasticity levels.")

    args = parser.parse_args()
    
    with open(args.config,'r') as f:
        config = json.load(f)
    
    t_max = config['t_max']
    num_cells = config['num_cells']
    num_realizations = config['num_realizations']
    root_folder = config['root_folder']
    
    params = aux.load_params(root_folder + 'simulation_params.json')

    if args.standard_pipeline:
        num_cells = 1000
        stims = build_standard_stims(t_max)
        sim_root = Path(root_folder) / "Simulations"
        run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = sim_root / f"{args.label}_{run_stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        stims = aux.load_stims(root_folder + 'stims.json')
        run_dir = Path(root_folder)
    
    xpt = sims.experiment(params,args.label)
    xpt.init_exp(num_cells)
    xpt.run_training_sim(stims, num_realizations)
    sim_file = xpt.save_cells(run_dir, standard_naming=False, custom_name=run_dir.name)
    if args.standard_pipeline:
        config_dump = {
            "circuit": args.label,
            "circuit_parameters": params,
            "simulation_file": str(sim_file),
            "simulated_cells": {
                "total_cells": num_cells,
                "random_stimulation_cells": [1, 900],
                "repetitive_stimulation_cells_red_first": [901, 950],
                "repetitive_stimulation_cells_green_first": [951, 1000],
                "num_realizations": num_realizations,
                "t_max": t_max,
            },
        }
        with open(run_dir / "config.json", "w") as f:
            json.dump(config_dump, f, indent=2)

    if args.standard_pipeline and args.noisy_sims > 0:
        temperatures = args.temperatures if args.temperatures is not None else [0.1]
        temp_schedule = assign_temperatures(args.noisy_sims, temperatures)
        noisy_root = run_dir / "noisy"
        noisy_root.mkdir(parents=True, exist_ok=True)

        for i, temp in enumerate(temp_schedule, start=1):
            sampled_params = aux.sample_noisy_params_from_dict(params, temperature=temp)
            sim_dir = noisy_root / f"sim_{i}"
            sim_dir.mkdir(parents=True, exist_ok=True)

            noisy_xpt = sims.experiment(sampled_params, args.label)
            noisy_xpt.init_exp(1000)
            noisy_xpt.run_training_sim(stims, num_realizations)
            noisy_file = noisy_xpt.save_cells(sim_dir, standard_naming=False, custom_name=sim_dir.name)

            noisy_config = {
                "circuit": args.label,
                "temperature": temp,
                "circuit_parameters": sampled_params,
                "simulation_file": str(noisy_file),
                "simulated_cells": {
                    "total_cells": 1000,
                    "random_stimulation_cells": [1, 900],
                    "repetitive_stimulation_cells_red_first": [901, 950],
                    "repetitive_stimulation_cells_green_first": [951, 1000],
                    "num_realizations": num_realizations,
                    "t_max": t_max,
                },
            }
            with open(sim_dir / "config.json", "w") as f:
                json.dump(noisy_config, f, indent=2)
