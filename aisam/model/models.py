import gillespy2
import numpy as np
from types import SimpleNamespace
from scipy import integrate


def add_multiplicative_noise(arrays, mean=1.0, sd=0.05, seed=None):
    rng = np.random.default_rng(seed)

    return [
        arr * rng.normal(loc=mean, scale=sd, size=arr.shape)
        for arr in arrays
    ]


def _nonnegative(value):
    return np.maximum(value, 0.0)


def _clean_ode_trace(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    cleaned = np.clip(values, 0.0, None)
    cleaned[~finite] = np.nan
    return cleaned


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
        self.reporter_species = None
        self.input_species = None
        self.time = 0
        self.events = []
        
        
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

class CcaSR(RXN):
    def __init__(self,params):
        super().__init__(name='CcaSR')
        self.params = params
        self.t_max = params['t_max']
        self.sampling = params['sampling']

    
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
   
    
    
class CcaSR_Inverter(RXN):
    def __init__(self,params):
        super().__init__(name='CcaSR_Inverter')
        self.params = params
        self.t_max = params['t_max']
        self.sampling = params['sampling']

    
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


class CcaSR_double_Inverter(RXN):
    def __init__(self,params):
        super().__init__(name='CcaSR_double_Inverter')
        self.params = params
        self.t_max = params['t_max']
        self.sampling = params['sampling']

    
    def init_rxn(self):
        params = SimpleNamespace(**self.params)
        alpha = params.alpha 
        beta = params.beta
        gamma = params.gamma
        k_tet = params.k_tet 
        k_lac = params.k_lac 
        k = params.k 
        n = params.n 
        n_tet = params.n_tet
        n_lac = params.n_lac
        tau_delay = params.tau_delay
        h1 = params.h1 
        h2 = params.h2 
        c2 = params.c2
        delta = params.delta

        alpha_g = gillespy2.Parameter(name='alpha', expression=alpha)
        beta_g = gillespy2.Parameter(name='beta', expression=beta)
        gamma_g = gillespy2.Parameter(name='gamma', expression=gamma)
        delta_g = gillespy2.Parameter(name='delta', expression=delta)
        c2_g = gillespy2.Parameter(name='c2', expression=c2)
        k_g = gillespy2.Parameter(name='k', expression=k)
        k_tet_g = gillespy2.Parameter(name='k_tet', expression=k_tet)
        n_g = gillespy2.Parameter(name='n', expression=n)
        n_tet_g = gillespy2.Parameter(name='n_tet', expression=n_tet)
        k_lac_g = gillespy2.Parameter(name='k_lac', expression=k_lac)
        n_lac_g = gillespy2.Parameter(name='n_lac', expression=n_lac)
        tau_delay_g = gillespy2.Parameter(name='tau_delay', expression=tau_delay)
        h1_g = gillespy2.Parameter(name='h1', expression=h1)
        h2_g = gillespy2.Parameter(name='h2', expression=h2)
        unit = gillespy2.Parameter(name = 'unit', expression=1)
        self.add_parameter([alpha_g,beta_g,gamma_g,delta_g,c2_g,k_g,k_tet_g,k_lac_g,n_g,n_tet_g,n_lac_g,tau_delay_g,h1_g,h2_g,unit])

        
        U = gillespy2.Species(name='U', initial_value=0,mode='discrete')
        H = gillespy2.Species(name='H', initial_value=0,mode='discrete')
        E = gillespy2.Species(name='E',   initial_value=round(np.random.poisson(h1/h2)),mode='discrete')
        T = gillespy2.Species(name='T',   initial_value=0,mode='discrete')
        L = gillespy2.Species(name='L',   initial_value=0,mode='discrete')
        F = gillespy2.Species(name='F',   initial_value=0,mode='discrete')
        
        self.reporter_species = 'F'
        self.input_species = 'U'
        self.add_species([U,H,E,T,L,F])
        
        # The list of reactants and products for a Reaction object are each a
        # Python dictionary in which the dictionary keys are Species objects
        # and the values are stoichiometries of the species in the reaction.
        H_c = gillespy2.Reaction(name="H_creation", propensity_function= 'U', reactants={}, products={H:1})
        H_d = gillespy2.Reaction(name="H_dissociation", propensity_function='c2*H', reactants={H:1}, products={})
        E_c = gillespy2.Reaction(name="E_creation", propensity_function= 'h1', reactants={}, products={E:1})
        E_d = gillespy2.Reaction(name="E_dissociation", propensity_function='h2*E', reactants={E:1}, products={})
        T_c = gillespy2.Reaction(name="T_creation", propensity_function= 'alpha*E*(pow(c2*H,n)/(k+pow(c2*H,n)))', reactants={}, products={T:1})
        T_d = gillespy2.Reaction(name="T_dissociation", propensity_function='delta*T', reactants={T:1}, products={})
        L_c = gillespy2.Reaction(name="L_creation", propensity_function= 'beta*E*(unit/(unit+pow(T/k_tet,n_tet)))', reactants={}, products={L:1})
        L_d = gillespy2.Reaction(name="L_dissociation", propensity_function='delta*L', reactants={L:1}, products={})
        F_c = gillespy2.Reaction(name="F_creation", propensity_function= 'gamma*E*(unit/(unit+pow(L/k_lac,n_lac)))', reactants={}, products={F:1})
        F_d = gillespy2.Reaction(name="F_dissociation", propensity_function='delta*F', reactants={F:1}, products={})
    
        self.add_reaction([H_c,H_d,E_c,E_d,T_c,T_d,L_c,L_d,F_c,F_d])





class CcaSR_noE(RXN):
    def __init__(self,params):
        super().__init__(name='CcaSR_noE')
        self.params = params
        self.t_max = params['t_max']
        self.sampling = params['sampling']

    
    def init_rxn(self):
        params = SimpleNamespace(**self.params)
        alpha = params.alpha 
        k = params.k 
        n = params.n 
        tau_delay = params.tau_delay
        c2 = params.c2
        delta = params.delta

        alpha_g = gillespy2.Parameter(name='alpha', expression=alpha)
        delta_g = gillespy2.Parameter(name='delta', expression=delta)
        c2_g = gillespy2.Parameter(name='c2', expression=c2)
        k_g = gillespy2.Parameter(name='k', expression=k)
        n_g = gillespy2.Parameter(name='n', expression=n)
        tau_delay_g = gillespy2.Parameter(name='tau_delay', expression=tau_delay)
        
        self.add_parameter([alpha_g,delta_g,c2_g,k_g,n_g,tau_delay_g])

        
        U = gillespy2.Species(name='U', initial_value=0,mode='discrete')
        H = gillespy2.Species(name='H', initial_value=0,mode='discrete')
        F = gillespy2.Species(name='F',   initial_value=0,mode='discrete')
        self.reporter_species = 'F'
        self.input_species = 'U'
        self.add_species([U,H,F])

        # The list of reactants and products for a Reaction object are each a
        # Python dictionary in which the dictionary keys are Species objects
        # and the values are stoichiometries of the species in the reaction.
        H_c = gillespy2.Reaction(name="H_creation", propensity_function= 'U', reactants={}, products={H:1})
        H_d = gillespy2.Reaction(name="H_dissociation", propensity_function='c2*H', reactants={H:1}, products={})
        F_c = gillespy2.Reaction(name="F_creation", propensity_function= 'alpha*(pow(c2*H,n)/(k+pow(c2*H,n)))', reactants={}, products={F:1})
        F_d = gillespy2.Reaction(name="F_dissociation", propensity_function='delta*F', reactants={F:1}, products={})
    
        self.add_reaction([H_c,H_d,F_c,F_d])
   
    
    
class CcaSR_Inverter_noE(RXN):
    def __init__(self,params):
        super().__init__(name='CcaSR_Inverter_noE')
        self.params = params
        self.t_max = params['t_max']
        self.sampling = params['sampling']

    
    def init_rxn(self):
        params = SimpleNamespace(**self.params)
        alpha = params.alpha 
        beta = params.beta
        k_tet = params.k_tet 
        k = params.k 
        n = params.n 
        n_tet = params.n_tet
        tau_delay = params.tau_delay
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
        unit = gillespy2.Parameter(name = 'unit', expression=1)
        self.add_parameter([alpha_g,beta_g,delta_g,c2_g,k_g,k_tet_g,n_g,n_tet_g,tau_delay_g,unit])

        
        U = gillespy2.Species(name='U', initial_value=0,mode='discrete')
        H = gillespy2.Species(name='H', initial_value=0,mode='discrete')
        T = gillespy2.Species(name='T',   initial_value=0,mode='discrete')
        F = gillespy2.Species(name='F',   initial_value=0,mode='discrete')
        
        self.reporter_species = 'F'
        self.input_species = 'U'
        self.add_species([U,H,T,F])
        
        # The list of reactants and products for a Reaction object are each a
        # Python dictionary in which the dictionary keys are Species objects
        # and the values are stoichiometries of the species in the reaction.
        H_c = gillespy2.Reaction(name="H_creation", propensity_function= 'U', reactants={}, products={H:1})
        H_d = gillespy2.Reaction(name="H_dissociation", propensity_function='c2*H', reactants={H:1}, products={})
        T_c = gillespy2.Reaction(name="T_creation", propensity_function= 'alpha*(pow(c2*H,n)/(k+pow(c2*H,n)))', reactants={}, products={T:1})
        T_d = gillespy2.Reaction(name="T_dissociation", propensity_function='delta*T', reactants={T:1}, products={})
        F_c = gillespy2.Reaction(name="F_creation", propensity_function= 'beta*(unit/(unit+pow(T/k_tet,n_tet)))', reactants={}, products={F:1})
        F_d = gillespy2.Reaction(name="F_dissociation", propensity_function='delta*F', reactants={F:1}, products={})
    
        self.add_reaction([H_c,H_d,T_c,T_d,F_c,F_d])
        


class CcaSR_double_Inverter_noE(RXN):
    def __init__(self,params):
        super().__init__(name='CcaSR_double_Inverter_noE')
        self.params = params
        self.t_max = params['t_max']
        self.sampling = params['sampling']

    
    def init_rxn(self):
        params = SimpleNamespace(**self.params)
        alpha = params.alpha 
        beta = params.beta
        gamma = params.gamma
        k_tet = params.k_tet 
        k_lac = params.k_lac 
        k = params.k 
        n = params.n 
        n_tet = params.n_tet
        n_lac = params.n_lac
        tau_delay = params.tau_delay
        c2 = params.c2
        delta = params.delta

        alpha_g = gillespy2.Parameter(name='alpha', expression=alpha)
        beta_g = gillespy2.Parameter(name='beta', expression=beta)
        gamma_g = gillespy2.Parameter(name='gamma', expression=gamma)
        delta_g = gillespy2.Parameter(name='delta', expression=delta)
        c2_g = gillespy2.Parameter(name='c2', expression=c2)
        k_g = gillespy2.Parameter(name='k', expression=k)
        k_tet_g = gillespy2.Parameter(name='k_tet', expression=k_tet)
        n_g = gillespy2.Parameter(name='n', expression=n)
        n_tet_g = gillespy2.Parameter(name='n_tet', expression=n_tet)
        k_lac_g = gillespy2.Parameter(name='k_lac', expression=k_lac)
        n_lac_g = gillespy2.Parameter(name='n_lac', expression=n_lac)
        tau_delay_g = gillespy2.Parameter(name='tau_delay', expression=tau_delay)
        unit = gillespy2.Parameter(name = 'unit', expression=1)
        self.add_parameter([alpha_g,beta_g,gamma_g,delta_g,c2_g,k_g,k_tet_g,k_lac_g,n_g,n_tet_g,n_lac_g,tau_delay_g,unit])

        
        U = gillespy2.Species(name='U', initial_value=0,mode='discrete')
        H = gillespy2.Species(name='H', initial_value=0,mode='discrete')
        T = gillespy2.Species(name='T',   initial_value=0,mode='discrete')
        L = gillespy2.Species(name='L',   initial_value=0,mode='discrete')
        F = gillespy2.Species(name='F',   initial_value=0,mode='discrete')
        
        self.reporter_species = 'F'
        self.input_species = 'U'
        self.add_species([U,H,T,L,F])
        
        # The list of reactants and products for a Reaction object are each a
        # Python dictionary in which the dictionary keys are Species objects
        # and the values are stoichiometries of the species in the reaction.
        H_c = gillespy2.Reaction(name="H_creation", propensity_function= 'U', reactants={}, products={H:1})
        H_d = gillespy2.Reaction(name="H_dissociation", propensity_function='c2*H', reactants={H:1}, products={})
        T_c = gillespy2.Reaction(name="T_creation", propensity_function= 'alpha*(pow(c2*H,n)/(k+pow(c2*H,n)))', reactants={}, products={T:1})
        T_d = gillespy2.Reaction(name="T_dissociation", propensity_function='delta*T', reactants={T:1}, products={})
        L_c = gillespy2.Reaction(name="L_creation", propensity_function= 'beta*(unit/(unit+pow(T/k_tet,n_tet)))', reactants={}, products={L:1})
        L_d = gillespy2.Reaction(name="L_dissociation", propensity_function='delta*L', reactants={L:1}, products={})
        F_c = gillespy2.Reaction(name="F_creation", propensity_function= 'gamma*(unit/(unit+pow(L/k_lac,n_lac)))', reactants={}, products={F:1})
        F_d = gillespy2.Reaction(name="F_dissociation", propensity_function='delta*F', reactants={F:1}, products={})
    
        self.add_reaction([H_c,H_d,T_c,T_d,L_c,L_d,F_c,F_d])

        
        
        
   

class ODE_CcaSR():
    def __init__(self,params,t_max=None,sampling=None,measurement_noise=None,std=None,x0=None):
        self.species = []
        self.reporter_species = None
        self.input_species = None
        self.params = dict(params)
        self.t_max = int(t_max if t_max is not None else self.params["t_max"])
        self.sampling = int(sampling if sampling is not None else self.params.get("sampling", 10))
        self.measurement_noise = bool(
            self.params.get("measurement_noise", False)
            if measurement_noise is None
            else measurement_noise
        )
        self.std = std if std is not None else self.params.get("std", 0.05)
        self.results = None
        self.x0 = list(x0 if x0 is not None else self.params.get("x0", [0, 0]))
        
    def init_rxn(self):
        params = SimpleNamespace(**self.params)
        self.alpha = params.alpha 
        self.k = params.k 
        self.n = params.n 
        self.tau_delay = params.tau_delay
        self.c2 = params.c2
        self.delta = params.delta
        
        self.species = ['H', 'F']
        self.reporter_species = 'F'
        self.input_species = 'U'
        
        self.results = {f'{specie}':[] for specie in self.species}
        self.initial_values = {f'{specie}':self.x0[i] for i,specie in enumerate(self.species)}

    def get_all_species(self):
        return {species: None for species in self.species}

    def rxn(self,x,t,U):
        H,F = x
        H_eff = _nonnegative(H)
        F_eff = _nonnegative(F)
        ind_t = min(max(0,int(np.floor((t-self.tau_delay)/5))), len(U)-1)
        H_signal = (self.c2 * H_eff) ** self.n
        dHdt = U[ind_t] - self.c2 * H_eff
        dFdt = self.alpha * H_signal / (self.k + H_signal) - self.delta * F_eff
        return [dHdt,dFdt] 
        
    def solve(self,stim,x0):
        t = np.linspace(0,self.t_max,max(2, int(self.t_max*self.sampling)),endpoint=True)
        x = integrate.odeint(self.rxn,x0,t,args=(stim,))
        
        run_results = {}
        for index , species in enumerate(self.species):
            run_results[species] = _clean_ode_trace(x[:,index])

        if self.measurement_noise:
            run_results[self.reporter_species] = add_multiplicative_noise(
                [run_results[self.reporter_species]],
                sd=self.std,
            )[0]
            run_results[self.reporter_species] = _clean_ode_trace(run_results[self.reporter_species])
            
        self.results = run_results
        return run_results
    
    def run_multi_rxn(self,stim_vec,num_trajectories = 1):
        if num_trajectories == 1:
            return self.solve(stim_vec,list(self.x0))
        return [self.solve(stim_vec,list(self.x0)) for _ in range(num_trajectories)]


class ODE_CcaSR_Inverter():
    def __init__(self,params,t_max=None,sampling=None,measurement_noise=None,std=None, x0=None):
        self.species = []
        self.reporter_species = None
        self.input_species = None
        self.params = dict(params)
        self.t_max = int(t_max if t_max is not None else self.params["t_max"])
        self.sampling = int(sampling if sampling is not None else self.params.get("sampling", 10))
        self.measurement_noise = bool(
            self.params.get("measurement_noise", False)
            if measurement_noise is None
            else measurement_noise
        )
        self.std = std if std is not None else self.params.get("std", 0.05)
        self.results = None
        self.x0 = list(x0 if x0 is not None else self.params.get("x0", [0, 0, 0]))
        
    def init_rxn(self):
        params = SimpleNamespace(**self.params)
        self.alpha = params.alpha 
        self.beta = params.beta
        self.k_tet = params.k_tet 
        self.k = params.k 
        self.n = params.n 
        self.n_tet = params.n_tet
        self.tau_delay = params.tau_delay
        self.c2 = params.c2
        self.delta = params.delta
    
        self.species = ['H', 'T', 'F']
        self.reporter_species = 'F'
        self.input_species = 'U'
        
        self.results = {f'{specie}':[] for specie in self.species}
        self.initial_values = {f'{specie}':self.x0[i] for i,specie in enumerate(self.species)}
    
    def get_all_species(self):
        return {species: None for species in self.species}
    
    def rxn(self,x,t,U):
        H,T,F = x
        H_eff = _nonnegative(H)
        T_eff = _nonnegative(T)
        F_eff = _nonnegative(F)
        ind_t = min(max(0,int(np.floor((t-self.tau_delay)/5))), len(U)-1)
        H_signal = (self.c2 * H_eff) ** self.n
        T_signal = (T_eff / self.k_tet) ** self.n_tet
        dHdt = U[ind_t] - self.c2 * H_eff
        dTdt = self.alpha * H_signal / (self.k + H_signal) - self.delta * T_eff
        dFdt = self.beta / (1 + T_signal) - self.delta * F_eff
        return [dHdt,dTdt,dFdt] 
        
    def solve(self,stim,x0):
        t = np.linspace(0,self.t_max,max(2, int(self.t_max*self.sampling)),endpoint=True)
        x = integrate.odeint(self.rxn,x0,t,args=(stim,))
        
        run_results = {}
        for index , species in enumerate(self.species):
            run_results[species] = _clean_ode_trace(x[:,index])

        if self.measurement_noise:
            run_results[self.reporter_species] = add_multiplicative_noise(
                [run_results[self.reporter_species]],
                sd=self.std,
            )[0]
            run_results[self.reporter_species] = _clean_ode_trace(run_results[self.reporter_species])
            
        self.results = run_results
        return run_results

    def run_multi_rxn(self,stim_vec,num_trajectories = 1):
        if num_trajectories == 1:
            return self.solve(stim_vec,list(self.x0))
        return [self.solve(stim_vec,list(self.x0)) for _ in range(num_trajectories)]


class ODE_CcaSR_double_Inverter():
    def __init__(self,params,t_max=None,sampling=None,measurement_noise=None,std=None, x0=None):
        self.species = []
        self.reporter_species = None
        self.input_species = None
        self.params = dict(params)
        self.t_max = int(t_max if t_max is not None else self.params["t_max"])
        self.sampling = int(sampling if sampling is not None else self.params.get("sampling", 10))
        self.measurement_noise = bool(
            self.params.get("measurement_noise", False)
            if measurement_noise is None
            else measurement_noise
        )
        self.std = std if std is not None else self.params.get("std", 0.05)
        self.results = None
        self.x0 = list(x0 if x0 is not None else self.params.get("x0", [0, 0, 0,0]))
        
    def init_rxn(self):
        params = SimpleNamespace(**self.params)
        self.alpha = params.alpha 
        self.beta = params.beta
        self.gamma = params.gamma
        self.k_tet = params.k_tet 
        self.k_lac = params.k_lac 
        self.k = params.k 
        self.n = params.n 
        self.n_tet = params.n_tet
        self.n_lac = params.n_lac
        self.tau_delay = params.tau_delay
        self.c2 = params.c2
        self.delta = params.delta
    
        self.species = ['H', 'T', 'L','F']
        self.reporter_species = 'F'
        self.input_species = 'U'
        
        self.results = {f'{specie}':[] for specie in self.species}
        self.initial_values = {f'{specie}':self.x0[i] for i,specie in enumerate(self.species)}
    
    def get_all_species(self):
        return {species: None for species in self.species}
    
    def rxn(self,x,t,U):
        H,T,L,F = x
        H_eff = _nonnegative(H)
        T_eff = _nonnegative(T)
        L_eff = _nonnegative(L)
        F_eff = _nonnegative(F)
        ind_t = min(max(0,int(np.floor((t-self.tau_delay)/5))), len(U)-1)
        H_signal = (self.c2 * H_eff) ** self.n
        T_signal = (T_eff / self.k_tet) ** self.n_tet
        L_signal = (L_eff / self.k_lac) ** self.n_lac
        dHdt = U[ind_t] - self.c2 * H_eff
        dTdt = self.alpha * H_signal / (self.k + H_signal) - self.delta * T_eff
        dLdt = self.beta / (1 + T_signal) - self.delta * L_eff
        dFdt = self.gamma / (1 + L_signal) - self.delta * F_eff
        return [dHdt,dTdt,dLdt,dFdt] 
        
    def solve(self,stim,x0):
        t = np.linspace(0,self.t_max,max(2, int(self.t_max*self.sampling)),endpoint=True)
        x = integrate.odeint(self.rxn,x0,t,args=(stim,))
        
        run_results = {}
        for index , species in enumerate(self.species):
            run_results[species] = _clean_ode_trace(x[:,index])

        if self.measurement_noise:
            run_results[self.reporter_species] = add_multiplicative_noise(
                [run_results[self.reporter_species]],
                sd=self.std,
            )[0]
            run_results[self.reporter_species] = _clean_ode_trace(run_results[self.reporter_species])
            
        self.results = run_results
        return run_results

    def run_multi_rxn(self,stim_vec,num_trajectories = 1):
        if num_trajectories == 1:
            return self.solve(stim_vec,list(self.x0))
        return [self.solve(stim_vec,list(self.x0)) for _ in range(num_trajectories)]


ODE_Inverter_CcaSR = ODE_CcaSR_Inverter


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
     
