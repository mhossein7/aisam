import argparse
import json
from rosam.utils import aux
from rosam.model import sims


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gene expression simulation")
    
    parser.add_argument("--label", type=str, default="CcaSR",
                        help="Name of the simulation")
    parser.add_argument("--config", type=str, default="",
                        help="Address to the config.json file")

    args = parser.parse_args()
    
    with open(args.config,'r') as f:
        config = json.load(f)
    
    t_max = config['t_max']
    num_cells = config['num_cells']
    num_realizations = config['num_realizations']
    root_folder = config['root_folder']
    
    params = aux.load_params(root_folder + 'simulation_params.json')
    stims = aux.load_stims(root_folder + 'stims.json')
    
    xpt = sims.experiment(params,args.label)
    xpt.init_exp(num_cells)
    xpt.run_training_sim(stims)
    xpt.save_cells(root_folder)
