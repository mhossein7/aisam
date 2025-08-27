import models
import utils
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt


delta = 0.01
alpha = 1
beta = 1
k = 0.4851
k_tet = 100
h1 = 2.3435*0.0303
h2 = 0.0303
tau_delay = 12
n=3.6
n_tet = 2
c2 = 0.0631
params = [alpha , k , n , tau_delay , h1 , h2 ,c2 , delta]
params_tet = [alpha , beta, k_tet , k , n ,n_tet, tau_delay , h1 , h2 , c2,delta]
tot_stim_vec = []
t_max = 180
sampling = 10
mega_res = {f'cell {i+1}':[] for i in range(10)}
Models = {}

for i in range(10):
    model = models.CcaSR_Inverter(params_tet,t_max,sampling)
    model.init_rxn()
    Models[f'cell {i+1}'] = model

for t in tqdm(range(10)):
    stim_vec = [utils.repetitive_stim_maker(2,int(t_max/5))]
    for i in range(10):
        model = Models[f'cell {i+1}']
        new_state = None if t==0 else model.give_updates(mega_res[f'cell {i+1}'][-1])
        results , new_state = model.run_online_rxn(updates=new_state,stim_vec=stim_vec[0])
        mega_res[f'cell {i+1}'].append(results)
    tot_stim_vec.append(stim_vec)


fig,ax = plt.subplots(2,1,figsize = (6,4))

utils.plot_w_bckgrnd(mega_res,np.concatenate(tot_stim_vec),t_max,line_color= 'b',axes=True,ax_out=ax[0])
utils.plot_w_bckgrnd(mega_res,np.concatenate(tot_stim_vec),t_max,'T',line_color = 'r',axes=True,ax_out = ax[1])
plt.tight_layout()
plt.show()