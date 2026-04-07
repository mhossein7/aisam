import gillespy2
import numpy as np
from types import SimpleNamespace
from scipy import integrate

class Simple(gillespy2.Model):
    def __init__(self,params,t_max,stim_vec):
        super().__init__(name='Simple_rxn')
        self.params = params
        self.t_max = t_max
        self.stim_vec = stim_vec
        self.reporter_species = None
        self.input_species = None
    
    def init_rxn(self):
        delta = self.params

        delta_g = gillespy2.Parameter(name='delta', expression=delta)
        
        self.add_parameter([delta_g])

        
        A = gillespy2.Species(name='A', initial_value=100,mode='discrete')
        B = gillespy2.Species(name='B', initial_value=0,mode='discrete')
        self.reporter_species = 'A'
        self.input_species = 'B'
        self.add_species([A,B])

        # The list of reactants and products for a Reaction object are each a
        # Python dictionary in which the dictionary keys are Species objects
        # and the values are stoichiometries of the species in the reaction.
        A_d = gillespy2.Reaction(name="A_degradation", propensity_function= 'delta*A', reactants={A:1}, products={})
        A_c = gillespy2.Reaction(name="A_creation", propensity_function= 'B', reactants={}, products={A:1})
        self.add_reaction([A_d,A_c])
    
    def update_rxn(self):
        for i,stim in enumerate(self.stim_vec):
            ea = gillespy2.EventAssignment(variable=self.input_species, expression=stim)
            et = gillespy2.EventTrigger(expression='1',initial_value=False)
            e = gillespy2.Event(name=f'event{i}', trigger=et, assignments=[ea],delay=f'{5*i}')
            self.add_event([e])
        

    def run_rxn(self):
        tspan = gillespy2.TimeSpan.linspace(t=self.t_max, num_points=int(self.t_max*10))
        self.timespan(tspan)
        self.init_rxn()
        self.update_rxn()
        results = self.run(number_of_trajectories=10,algorithm="Tau-Hybrid")
        return results


class RXN(gillespy2.Model):
    def __init__(self,name = ""):
        super().__init__(name=name)
        
    def update_events(self,stim_vec):
        self.events.append(stim_vec)
    
    def update_rxn(self,stim_vec):
        self.delete_all_events()
        if len(self.events)!=0:
            for (i,leftover_delay) in list(zip([-2,-1],['2','7'])):
                ea = gillespy2.EventAssignment(variable=self.input_species, expression=self.events[-1][i])
                et = gillespy2.EventTrigger(expression='1',initial_value=False)
                e = gillespy2.Event(name=f'lf_event{np.abs(i)}', trigger=et, assignments=[ea],delay=leftover_delay)
                self.add_event([e])
        
        for i,stim in enumerate(stim_vec):
            ea = gillespy2.EventAssignment(variable=self.input_species, expression=stim)
            et = gillespy2.EventTrigger(expression='1',initial_value=False)
            e = gillespy2.Event(name=f'event{i}', trigger=et, assignments=[ea],delay=f'{(5*i)+self.params['tau_delay']}') 
            self.add_event([e])
        self.update_events(stim_vec)
    
    def get_updates(self,updates):
        self.delete_all_species()
        for name,value in updates.items():
            self.add_species(gillespy2.Species(name = name, initial_value= value, mode = 'discrete'))    

    def give_updates(self,results):
        updates = {}
        for name in self.get_all_species().keys():
                updates[name] = results[name][-1]
        return updates
    
    def run_online_rxn(self,updates,stim_vec):
        tspan = gillespy2.TimeSpan.linspace(t=self.t_max, num_points=int(self.t_max*self.sampling))
        self.timespan(tspan)
        if updates is not None:
            self.get_updates(updates)
        self.update_rxn(stim_vec)
        results = self.run(algorithm= 'Tau-Hybrid')
        new_state = self.give_updates(results)
        return results,new_state
    
    def run_multi_rxn(self,stim_vec,num_trajectories = 1):
        tspan = gillespy2.TimeSpan.linspace(t=self.t_max, num_points=int(self.t_max*self.sampling))
        self.timespan(tspan)
        self.update_rxn(stim_vec)
        results = self.run(number_of_trajectories = num_trajectories, algorithm= 'Tau-Hybrid')
        return results

class CcaSR(gillespy2.Model):
    def __init__(self,params):
        super().__init__(name='CcaSR')
        self.params = params
        self.t_max = params['t_max']
        self.sampling = params['sampling']
        self.reporter_species = None
        self.input_species = None
        self.time = 0
        self.events = []
    
    def init_rxn(self):
        params = SimpleNamespace(**self.params)
        alpha = params.alpha 
        k = params.k 
        n = params.n 
        tau_delay = params.tau_delay
        h1 = params.h1 
        h2 = params.h2 
        c2 = params.c2
        delta = params.delta

        alpha_g = gillespy2.Parameter(name='alpha', expression=alpha)
        delta_g = gillespy2.Parameter(name='delta', expression=delta)
        c2_g = gillespy2.Parameter(name='c2', expression=c2)
        k_g = gillespy2.Parameter(name='k', expression=k)
        n_g = gillespy2.Parameter(name='n', expression=n)
        tau_delay_g = gillespy2.Parameter(name='tau_delay', expression=tau_delay)
        h1_g = gillespy2.Parameter(name='h1', expression=h1)
        h2_g = gillespy2.Parameter(name='h2', expression=h2)
        
        self.add_parameter([alpha_g,delta_g,c2_g,k_g,n_g,tau_delay_g,h1_g,h2_g])

        
        U = gillespy2.Species(name='U', initial_value=0,mode='discrete')
        H = gillespy2.Species(name='H', initial_value=0,mode='discrete')
        E = gillespy2.Species(name='E',   initial_value=round(np.random.poisson(h1/h2)),mode='discrete')
        F = gillespy2.Species(name='F',   initial_value=0,mode='discrete')
        self.reporter_species = 'F'
        self.input_species = 'U'
        self.add_species([U,H,E,F])

        # The list of reactants and products for a Reaction object are each a
        # Python dictionary in which the dictionary keys are Species objects
        # and the values are stoichiometries of the species in the reaction.
        H_c = gillespy2.Reaction(name="H_creation", propensity_function= 'U', reactants={}, products={H:1})
        H_d = gillespy2.Reaction(name="H_dissociation", propensity_function='c2*H', reactants={H:1}, products={})
        E_c = gillespy2.Reaction(name="E_creation", propensity_function= 'h1', reactants={}, products={E:1})
        E_d = gillespy2.Reaction(name="E_dissociation", propensity_function='h2*E', reactants={E:1}, products={})
        F_c = gillespy2.Reaction(name="F_creation", propensity_function= 'alpha*E*(pow(c2*H,n)/(k+pow(c2*H,n)))', reactants={}, products={F:1})
        F_d = gillespy2.Reaction(name="F_dissociation", propensity_function='delta*F', reactants={F:1}, products={})
    
        self.add_reaction([H_c,H_d,E_c,E_d,F_c,F_d])
    
    def update_events(self,stim_vec):
        self.events.append(stim_vec)
    
    def update_rxn(self,stim_vec):
        self.delete_all_events()
        if len(self.events)!=0:
            for (i,leftover_delay) in list(zip([-2,-1],['2','7'])):
                ea = gillespy2.EventAssignment(variable=self.input_species, expression=self.events[-1][i])
                et = gillespy2.EventTrigger(expression='1',initial_value=False)
                e = gillespy2.Event(name=f'lf_event{np.abs(i)}', trigger=et, assignments=[ea],delay=leftover_delay)
                self.add_event([e])
                
        for i,stim in enumerate(stim_vec):
            ea = gillespy2.EventAssignment(variable=self.input_species, expression=stim)
            et = gillespy2.EventTrigger(expression='1',initial_value=False)
            e = gillespy2.Event(name=f'event{i}', trigger=et, assignments=[ea],delay=f'{(5*i)+self.params['tau_delay']}')
            self.add_event([e])
        self.update_events(stim_vec)
    
    def get_updates(self,updates):
        self.delete_all_species()
        for name,value in updates.items():
            self.add_species(gillespy2.Species(name = name, initial_value= value, mode = 'discrete'))    

    def give_updates(self,results):
        updates = {}
        for name in self.get_all_species().keys():
                updates[name] = results[name][-1]
        return updates
    
    def run_online_rxn(self,updates,stim_vec):
        tspan = gillespy2.TimeSpan.linspace(t=self.t_max, num_points=int(self.t_max*self.sampling))
        self.timespan(tspan)
        if updates is not None:
            self.get_updates(updates)
        self.update_rxn(stim_vec)
        results = self.run(number_of_trajectories = 1, algorithm= 'Tau-Hybrid')
        new_state = self.give_updates(results)
        return results,new_state
    
    
    
class CcaSR_Inverter(gillespy2.Model):
    def __init__(self,params):
        super().__init__(name='CcaSR_Inverter')
        self.params = params
        self.t_max = params['t_max']
        self.sampling = params['sampling']
        self.reporter_species = None
        self.input_species = None
        self.time = 0
        self.events = []
    
    def init_rxn(self):
        params = SimpleNamespace(**self.params)
        alpha = params.alpha 
        beta = params.beta
        k_tet = params.k_tet 
        k = params.k 
        n = params.n 
        n_tet = params.n_tet
        tau_delay = params.tau_delay
        h1 = params.h1 
        h2 = params.h2 
        c2 = params.c2
        delta = params.delta

        alpha_g = gillespy2.Parameter(name='alpha', expression=alpha)
        beta_g = gillespy2.Parameter(name='beta', expression=beta)
        delta_g = gillespy2.Parameter(name='delta', expression=delta)
        c2_g = gillespy2.Parameter(name='c2', expression=c2)
        k_g = gillespy2.Parameter(name='k', expression=k)
        k_tet_g = gillespy2.Parameter(name='k_tet', expression=k_tet)
        n_g = gillespy2.Parameter(name='n', expression=n)
        n_tet_g = gillespy2.Parameter(name='n_tet', expression=n_tet)
        tau_delay_g = gillespy2.Parameter(name='tau_delay', expression=tau_delay)
        h1_g = gillespy2.Parameter(name='h1', expression=h1)
        h2_g = gillespy2.Parameter(name='h2', expression=h2)
        unit = gillespy2.Parameter(name = 'unit', expression=1)
        self.add_parameter([alpha_g,beta_g,delta_g,c2_g,k_g,k_tet_g,n_g,n_tet_g,tau_delay_g,h1_g,h2_g,unit])

        
        U = gillespy2.Species(name='U', initial_value=0,mode='discrete')
        H = gillespy2.Species(name='H', initial_value=0,mode='discrete')
        E = gillespy2.Species(name='E',   initial_value=round(np.random.poisson(h1/h2)),mode='discrete')
        T = gillespy2.Species(name='T',   initial_value=0,mode='discrete')
        F = gillespy2.Species(name='F',   initial_value=0,mode='discrete')
        
        self.reporter_species = 'F'
        self.input_species = 'U'
        self.add_species([U,H,E,T,F])
        
        # The list of reactants and products for a Reaction object are each a
        # Python dictionary in which the dictionary keys are Species objects
        # and the values are stoichiometries of the species in the reaction.
        H_c = gillespy2.Reaction(name="H_creation", propensity_function= 'U', reactants={}, products={H:1})
        H_d = gillespy2.Reaction(name="H_dissociation", propensity_function='c2*H', reactants={H:1}, products={})
        E_c = gillespy2.Reaction(name="E_creation", propensity_function= 'h1', reactants={}, products={E:1})
        E_d = gillespy2.Reaction(name="E_dissociation", propensity_function='h2*E', reactants={E:1}, products={})
        T_c = gillespy2.Reaction(name="T_creation", propensity_function= 'alpha*E*(pow(c2*H,n)/(k+pow(c2*H,n)))', reactants={}, products={T:1})
        T_d = gillespy2.Reaction(name="T_dissociation", propensity_function='delta*T', reactants={T:1}, products={})
        F_c = gillespy2.Reaction(name="F_creation", propensity_function= 'beta*E*(unit/(unit+pow(T/k_tet,n_tet)))', reactants={}, products={F:1})
        F_d = gillespy2.Reaction(name="F_dissociation", propensity_function='delta*F', reactants={F:1}, products={})
    
        self.add_reaction([H_c,H_d,E_c,E_d,T_c,T_d,F_c,F_d])
        
        
    def update_events(self,stim_vec):
        self.events.append(stim_vec)
    
    def update_rxn(self,stim_vec):
        self.delete_all_events()
        if len(self.events)!=0:
            for (i,leftover_delay) in list(zip([-2,-1],['2','7'])):
                ea = gillespy2.EventAssignment(variable=self.input_species, expression=self.events[-1][i])
                et = gillespy2.EventTrigger(expression='1',initial_value=False)
                e = gillespy2.Event(name=f'lf_event{np.abs(i)}', trigger=et, assignments=[ea],delay=leftover_delay)
                self.add_event([e])
        
        for i,stim in enumerate(stim_vec):
            ea = gillespy2.EventAssignment(variable=self.input_species, expression=stim)
            et = gillespy2.EventTrigger(expression='1',initial_value=False)
            e = gillespy2.Event(name=f'event{i}', trigger=et, assignments=[ea],delay=f'{(5*i)+self.params['tau_delay']}') 
            self.add_event([e])
        self.update_events(stim_vec)
    
    def get_updates(self,updates):
        self.delete_all_species()
        for name,value in updates.items():
            self.add_species(gillespy2.Species(name = name, initial_value= value, mode = 'discrete'))    

    def give_updates(self,results):
        updates = {}
        for name in self.get_all_species().keys():
                updates[name] = results[name][-1]
        return updates
    
    def run_online_rxn(self,updates,stim_vec):
        tspan = gillespy2.TimeSpan.linspace(t=self.t_max, num_points=int(self.t_max*self.sampling))
        self.timespan(tspan)
        if updates is not None:
            self.get_updates(updates)
        self.update_rxn(stim_vec)
        results = self.run(algorithm= 'Tau-Hybrid')
        new_state = self.give_updates(results)
        return results,new_state
    
    def run_multi_rxn(self,stim_vec,num_trajectories = 1):
        tspan = gillespy2.TimeSpan.linspace(t=self.t_max, num_points=int(self.t_max*self.sampling))
        self.timespan(tspan)
        self.update_rxn(stim_vec)
        results = self.run(number_of_trajectories = num_trajectories, algorithm= 'Tau-Hybrid')
        return results
    
    

class ODE_CcaSR():
    def __init__(self,params,t_max,sampling = 10):
        self.params = params
        self.t_max = t_max
        self.sampling = sampling
        params = SimpleNamespace(**self.params)
        self.alpha = params.alpha 
        self.beta = params.beta
        self.k_tet = params.k_tet 
        self.k = params.k 
        self.n = params.n 
        self.n_tet = params.n_tet
        self.tau_delay = params.tau_delay
        self.h1 = params.h1 
        self.h2 = params.h2 
        self.c2 = params.c2
        self.delta = params.delta
        
    def rxn(self,x,t,U):
        H,F = x
        ind_t = min(max(0,int(np.floor((t-self.tau_delay)/5))), len(U)-1)
        dHdt = U[ind_t] - self.c2 * H
        dFdt = self.alpha * (self.c2*H)**self.n/(self.k + (self.c2*H)**self.n) - self.delta * F
        return [dHdt,dFdt] 
        
    def solve(self,stim,x0):
        t = np.linspace(0,self.t_max,self.t_max*self.sampling,endpoint=True)
        x = integrate.odeint(self.rxn,x0,t,args=(stim,))
        return x


class ODE_CcaSR_Inverter():
    def __init__(self,params,t_max,sampling = 10):
        self.params = params
        self.t_max = t_max
        self.sampling = sampling
        params = SimpleNamespace(**self.params)
        self.alpha = params.alpha 
        self.beta = params.beta
        self.k_tet = params.k_tet 
        self.k = params.k 
        self.n = params.n 
        self.n_tet = params.n_tet
        self.tau_delay = params.tau_delay
        self.h1 = params.h1 
        self.h2 = params.h2 
        self.c2 = params.c2
        self.delta = params.delta
        
    def rxn(self,x,t,U):
        H,T,F = x
        ind_t = min(max(0,int(np.floor((t-self.tau_delay)/5))), len(U)-1)
        dHdt = U[ind_t] - self.c2 * H
        dTdt = self.alpha * (self.c2*H)**self.n/(self.k + (self.c2*H)**self.n) - self.delta * T
        dFdt = self.beta/(1+(T/self.k_tet)**self.n_tet) - self.delta * F
        return [dHdt,dTdt,dFdt] 
        
    def solve(self,stim,x0):
        t = np.linspace(0,self.t_max,self.t_max*self.sampling,endpoint=True)
        x = integrate.odeint(self.rxn,x0,t,args=(stim,))
        return x



class Simple_spring_mass():
    def __init__(self,m,k,c,xr = 0,x0=0,v0=0,g=10,dt=0.1):
        self.m = m
        self. k = k
        self.c = c
        self.xr= xr
        self.v0 = v0
        self.x0 = x0
        self.g = g
        self.dt = dt
        
        self.x = [x0]
        self.v = [v0]
        self.U = [0]
        self.time = 0
        return
    
    def __str__(self):
        sampling = int(1/self.dt)
        if self.time < 5:
            i = int(np.floor(self.time))
            past_forces = np.array(self.U)[-1*np.arange(1,i*sampling,sampling)][::-1]
            past_positions = np.array(self.x)[-1*np.arange(1,i*sampling,sampling)][::-1]
            past_velocities = np.array(self.v)[-1*np.arange(1,i*sampling,sampling)][::-1]
        else: 
            i = 5
            past_forces = np.array(self.U)[-1*np.arange(1,i*sampling,sampling)][::-1]
            past_positions = np.array(self.x)[-1*np.arange(1,i*sampling,sampling)][::-1]
            past_velocities = np.array(self.v)[-1*np.arange(1,i*sampling,sampling)][::-1]
        return f'''
                        System description:
                            mass (m)= {self.m}
                            spring constant (k)= {self.k}
                            damper constant (c)= {self.c}
                            spring resting location (x_r)= {self.xr}
                            gravity constant (g) = {self.g}
                            current position (x) = {self.x[-1]}
                            current velocity (v) = {self.v[-1]}
                            current force (u) = {self.U[-1]}
                            current time (t) = {self.time} s
                            forces applied in the past {i} s = {past_forces}
                            position in the past {i} s = {past_positions}
                            velocity in the past {i} s = {past_velocities}
                        System dynamics: 
                            dv = dt/m *(-k*(x-x_r) - v*c + u + m*g) 
                            dx = dt*v
                            v[t+1] = v[t] + dv
                            x[t+1] = x[t] + dx
        '''
    def reset(self):
        self.x = [self.x0]
        self.v = [self.v0]
        self.U = [0]
        self.time = 0

    def set_(self,x,v,t,U):
        self.x = list(x)
        self.v = list(v)
        self.time = t
        self.U = list(U)

    def exert(self,u):
        self.U.append(u)
        dv = self.dt/self.m*(-self.k*(self.x[-1]-self.xr)-self.v[-1]*self.c+u+self.m*self.g)
        v_t = self.v[-1] + dv
        dx = v_t*self.dt
        x_t = self.x[-1] + dx
        self.x.append(x_t)
        self.v.append(v_t)
        self.time+= self.dt
    
    def control(self,x_g,p):
        #closed loop (proportional-integral)
        if self.x[-1]== x_g : return
        else: 
            hist = self.x
            errors = [x_g - x for x in hist]
            integral = np.sum(errors)*self.dt
            self.exert(p*integral)
            return p*integral
    
    def state(self):
        return self.x,self.v
     