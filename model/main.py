import models
import utils
from tqdm import tqdm

import numpy as np


delta = 0.01
alpha = 1
k = 0.4851
h1 = 2.3435*0.0303
h2 = 0.0303
tau_delay = 12
n=3.6
c2 = 0.0631
params = [alpha , k , n , tau_delay , h1 , h2 ,c2 , delta]
tot_stim_vec = []
t_max = 120
sampling = 10
mega_res = {f'cell {i+1}':[] for i in range(10)}
Models = {}

for i in range(10):
    model = models.CcaSR(params,t_max,sampling)
    model.init_rxn()
    Models[f'cell {i+1}'] = model

for t in tqdm(range(8)):
    stim_vec = [utils.repetitive_stim_maker(np.random.choice(np.arange(1,25),1),int(t_max/5))]
    for i in range(10):
        model = Models[f'cell {i+1}']
        new_state = None if t==0 else model.give_updates(mega_res[f'cell {i+1}'][-1])
        results , new_state = model.run_online_rxn(updates=new_state,stim_vec=stim_vec[0])
        mega_res[f'cell {i+1}'].append(results)
    tot_stim_vec.append(stim_vec)


utils.plot_w_bckgrnd(mega_res,np.concatenate(tot_stim_vec),t_max)
