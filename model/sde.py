import numpy as np
import matplotlib.pyplot as plt



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
stim_vec = np.random.choice(2,num_steps, p =[0.55,.45])
stim_vec = np.hstack((ons,offs,ons,offs))
X_0 = [0, np.round(np.random.poisson(h1/h2)),0] 


t_max = 1000

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
        lambda x,t: stim_vec[np.max(int(np.floor((t-tau_delay)/5)),int(0))],
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
        if a0 == 0:
            break  # No reactions can occur

        # Time to next reaction
        tau = np.random.exponential(1 / a0)
        t += tau

        # Choose reaction
        r = np.random.uniform(0, a0)
        j = np.searchsorted(np.cumsum(a), r)

        # Update species
        X += sto_mat[j]
        Outputs = X.copy()
        #Outputs[2] = Outputs[2] + np.random.normal(0,(0.05*Outputs[2]))
        # Record
        time.append(t)
        history.append(Outputs.copy())

    return np.array(time), np.array(history)



fig, ax = plt.subplots(figsize=(8, 4))
for i, val in enumerate(stim_vec):
    x_start = i * 5
    x_end = x_start + 5
    color = 'green' if val == 1 else 'red'
    ax.axvspan(x_start, x_end, facecolor=color, alpha=0.2)


t_max = 1000
time, X_vals = gillespie_CcaSR(X_0, params, stim_vec, t_max)
ax.step(time,X_vals[:,2], where='post')


plt.xlabel("Time")
plt.ylabel("X (molecule count)")
plt.title("Gillespie Simulation of Birth-Death Process")
ax.grid(True)
plt.tight_layout()
plt.show()
