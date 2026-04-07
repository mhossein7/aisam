import numpy as np
import rosam
from rosam.model import models
from rosam.utils import aux
import json
import os
from tqdm import tqdm
import copy 
import gillespy2
import dill
from pathlib import Path
from datetime import datetime

class Cell_sim():
    def __init__(self,circuit:str):
        self.stims = []
        self.species = []
        self.circuit_params = {}
        self.circuit = circuit
        self.expressions = {}
        self.model = None
        
        
    def assign_model(self):
        if not self.circuit_params: raise Exception("Please first assign parameters to the Cell")
        if self.circuit.lower() == 'ccasr':
            circuit_model = models.CcaSR(self.circuit_params)
        elif self.circuit.lower() == 'inverter':
            circuit_model = models.CcaSR_Inverter(self.circuit_params)
        else: 
            raise ValueError("Model not implemented yet")
        
        self.model = copy.deepcopy(circuit_model)
        
    
    def assign_parameters(self,params):
        if isinstance(params,dict):
            self.circuit_params = params
        elif isinstance(params,str):
            if os.path.isfile(params) and params.lower().endswith(".json"):
                try:
                    with open(params, "r") as f:
                        self.circuit_params = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Error decoding JSON file '{params}': {e}")
            else:
                raise ValueError(f"'{params}' is not a valid JSON file path.")
        else:
            raise ValueError("Parameter input must be either a dictionary or a JSON file path string.")
    
    def assign_features(self):
        for species in self.model.get_all_species().keys():
            self.species.append(species)
        for species in self.species:
            self.expressions[species] = []
        
        
    def assign_run(self,stim_vec,results,realizations = 1):
        self.stims.append(stim_vec)
        for species in self.species:
            if realizations >1:
                expressions = []
                for i in range(realizations):
                    expressions.append(results[i][species])
                expressions = np.array(expressions)
                self.expressions[species].append(expressions)
            else: self.expressions[species].append(results[species])
    
    
    
        

class experiment():
    def __init__(self,params,circuit:str = 'CcaSR'):
        self.circuit = circuit
        self.Cells = None
        self.params = params
        
    def init_exp(self,num_cells):
        self.Cells = {}
           
        for i in range(num_cells):
            params = self.params[i] if isinstance(self.params,list) else self.params
            self.Cells[f'{i+1}'] = Cell_sim(self.circuit)
            self.Cells[f'{i+1}'].assign_parameters(params)
            self.Cells[f'{i+1}'].assign_model()
            self.Cells[f'{i+1}'].model.init_rxn()
            self.Cells[f'{i+1}'].assign_features()
            
    def run_training_sim(self,stims,num_realizations):
        num_cells = len(self.Cells.items())
        for i in range(num_cells):
            stim_vec = stims[f'cell {i+1}']
            model = self.Cells[f'{i+1}'].model
            results = model.run_multi_rxn(stim_vec=stim_vec,num_trajectories = num_realizations)
            self.Cells[f'{i+1}'].assign_run(stim_vec,results,num_realizations)
                
    def save_cells(self,root_address,standard_naming=True,custom_name = ""):
        label = self.circuit
        if standard_naming:
            date_str = datetime.now().strftime("%Y-%m-%d")
            
            address = Path(root_address) / f"{label}_{date_str}" 
            address.mkdir(parents=True, exist_ok=True)
        
            time_str = datetime.now().strftime("%H-%M-%S")
            file_path = address / f"simulation_{time_str}.pkl"       
        else:
            address = Path(root_address) / f'{label}_{custom_name}'
            address.mkdir(parents=True, exist_ok=True)
            file_path = address / f"simulation_{time_str}.pkl" 
            
        with open(file_path, "wb") as f:
            dill.dump(self.Cells, f)


