import numpy as np
import matplotlib.pyplot as plt

def rep_stim_maker(num_repeat,total_time):
    num_reps = int(np.floor(total_time/num_repeat))
    ons = np.repeat(1,num_reps)
    offs = np.repeat(0,num_reps)
    tile = np.hstack((ons,offs))
    stim_vec = np.tile(tile,int(num_repeat))
    stim_vec = stim_vec[:total_time]
    return stim_vec


delta = 0.01
alpha = 1
k = 0.4851
h1 = 2.3435*0.0303
h2 = 0.0303
tau_delay = 12
n=3.6
c2 = 0.0631
params = [alpha , k , n , tau_delay , h1 , h2 ,c2 , delta]

num_steps = 200
ons = np.repeat(1,50)
offs = np.repeat(0,50)
#stim_vec = np.random.choice(2,num_steps, p =[0.75,0.25])
stim_vec = rep_stim_maker(10,80)
#stim_vec2 = np.hstack((stim_vec,np.zeros(100),np.ones(50),np.zeros(50)))
X_0 = [0, round(np.random.poisson(h1/h2)),0] 


def gillespie_CcaSR(init_con,params,stim_vec, t_max):
    # Initialize
    t = 0
    alpha , k , n , tau_delay , h1 , h2 , c2,delta = params
    
    sto_mat = np.array([
    [1,0,0],
    [-1,0,0],
    [0,1,0],
    [0,-1,0],
    [0,0,1],
    [0,0,-1]
    ])

    prop_funs = [
        lambda x,t: stim_vec[np.max(int(np.floor((t/10-tau_delay)/5)),int(0))],
        lambda x,t: delta*x[0],
        lambda x,t: h1,
        lambda x,t: h2,
        lambda x,t: alpha * x[1] * ((c2*x[0])**n/(k + (c2*x[0])**n)),
        lambda x,t: delta * x[2]
    ]

    t = 0.0
    X = init_con.copy()
    time = [t]
    history = [X.copy()]

    while t < t_max:
        a = np.array([f(X,t) for f in prop_funs])
        a0 = a.sum()
        if a0 <= 1e-2:
            break  
        
        
        tau = np.random.exponential(np.abs(1 / a0))
        t += tau
        if t > t_max: break

        # Choose reaction
        r = np.random.uniform(0, a0)
        j = np.searchsorted(np.cumsum(a), r)

        # Update species
        X += sto_mat[j]
        Outputs = X.copy()/100
        #Outputs[2] = Outputs[2] + np.random.normal(0,(0.0005*Outputs[2]))
        # Record
        time.append(t)
        history.append(Outputs.copy())

    return np.array(time), np.array(history)



fig, ax = plt.subplots(figsize=(8, 4))
for i, val in enumerate(stim_vec):
    x_start = i * 50
    x_end = x_start + 50
    color = 'green' if val == 1 else 'red'
    ax.axvspan(x_start, x_end, facecolor=color, alpha=0.2)


t_max = 4000
Xs = []
lens = []
for i in range(50):
    time, X_vals = gillespie_CcaSR(X_0, params, stim_vec, t_max)
    Xs.append(X_vals[:,2])
    ax.step(time,Xs[i], where='post',color = 'blue',alpha= 0.5)
    lens.append(len(X_vals[:,2]))
#Xs = np.array(Xs)
#X_mean = np.mean(Xs,axis = 0)



ax.set_xticks(np.arange(0,4500,500))
ax.set_xticklabels(np.arange(0,450,50))
ax.set_xlabel('Time (min)')
ax.set_ylabel('GFP (molecule count)')
ax.grid(False)
plt.tight_layout()
plt.show()
