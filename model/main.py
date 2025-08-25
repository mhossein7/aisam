import models
import utils

import numpy as np

stim_vec2 = utils.repetitive_stim_maker(2,80)

delta = 0.01
alpha = 1
k = 0.4851
h1 = 2.3435*0.0303
h2 = 0.0303
tau_delay = 12
n=3.6
c2 = 0.0631
params = [alpha , k , n , tau_delay , h1 , h2 ,c2 , delta]
#stim_vec = np.random.choice(2,80)
stim_vec =[stim_vec2,stim_vec2[::-1],stim_vec2]
t_max = 400
sampling = 10
mega_res = {f'cell {i+1}':[] for i in range(10)}
Models = {}

for i in range(10):
    model = models.CcaSR(params,t_max,sampling)
    model.init_rxn()
    Models[f'cell {i+1}'] = model

for i in range(10):
    model = Models[f'cell {i+1}']
    new_state = None
    for j,stim in enumerate(stim_vec):
        results , new_state = model.run_online_rxn(updates=new_state,stim_vec=stim)
        mega_res[f'cell {i+1}'].append(results)

for i in range(10):
    model = Models[f'cell {i+1}']
    new_state = model.give_updates(mega_res[f'cell {i+1}'][-1])
    for j,stim in enumerate(stim_vec):
        results , new_state = model.run_online_rxn(updates=new_state,stim_vec=stim)
        mega_res[f'cell {i+1}'].append(results)


utils.plot_w_bckgrnd(mega_res,np.concatenate(stim_vec,stim_vec),t_max)