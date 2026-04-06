import numpy as np
from matplotlib import pyplot as plt


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



def plot_w_bckgrnd(mega_res,stim_vec_tot,t_max,species = 'F',sampling=10,line_color= 'b',save=None,axes = False,ax_out=None):
    '''
    mega_res: dictionary with all cell simulation results
    stim_vec_tot: list of arrays of stims in the form of 1 and 0 for the whole history of simulation
    t_max: total time of each simulation 
    sampling: data sampling during simulation (default = 10) 
    save: a dictionary with information about saving the figure. Default is None which causes only showing the figure
    '''
    if ax_out is not None: 
        ax = ax_out
        
    else: 
        fig, ax = plt.subplots(figsize=(8, 4))
    
    for j in range(len(stim_vec_tot)):
        for i, val in enumerate(stim_vec_tot[j]):
            x_start = j*t_max*sampling + i * 5*sampling
            x_end = x_start + 5*sampling
            color = 'green' if val == 1 else 'red'
            ax.axvspan(x_start, x_end, facecolor=color, alpha=0.2)
    for k in range(len(mega_res.keys())):
        results = mega_res[f'cell {k+1}']
        for i in range(len(results)):
            ax.plot(np.arange(t_max*i*sampling,t_max*(i+1)*sampling),(results[i][species]),color=line_color,linewidth = 0.5)

    if save is not None:
        path = save['path']
        fig.savefig(path,dpi=300,format = 'svg')
        
    elif axes == True:
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('GFP (molecule count)')
        return ax
    
    else:    
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('GFP (molecule count)')
        plt.tight_layout()
        plt.show()  
        
        
def background_plotter(ax,stim_vec,sampling = 10, stim_period=5):
    for i, val in enumerate(stim_vec):
            x_start =  sampling*i * stim_period
            x_end = x_start + sampling*stim_period
            color = 'green' if val == 1 else 'red'
            ax.axvspan(x_start, x_end, facecolor=color, alpha=0.2)