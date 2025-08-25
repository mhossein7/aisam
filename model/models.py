import gillespy2
import numpy as np

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


class CcaSR(gillespy2.Model):
    def __init__(self,params,t_max,sampling=10):
        super().__init__(name='CcaSR')
        self.params = params
        self.t_max = t_max
        self.sampling = sampling
        self.reporter_species = None
        self.input_species = None
    
    def init_rxn(self):
        alpha , k , n , tau_delay , h1 , h2 , c2,delta = self.params

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
    
    def update_rxn(self,stim_vec):
        self.delete_all_events()
        for i,stim in enumerate(stim_vec):
            ea = gillespy2.EventAssignment(variable=self.input_species, expression=stim)
            et = gillespy2.EventTrigger(expression='1',initial_value=False)
            e = gillespy2.Event(name=f'event{i}', trigger=et, assignments=[ea],delay=f'{(5*i)+self.params[3]}')
            self.add_event([e])
    
    def get_updates(self,updates):
        self.delete_all_species()
        for name,value in updates.items():
            self.add_species(gillespy2.Species(name = name, initial_value= value, mode = 'discrete'))    

    def give_updates(self,results):
        updates = {}
        for name in self.get_all_species().keys():
                updates[name] = results[name][-1]
        return updates
    
    def run_rxn(self,stim_vec):
        tot_results = []
        tspan = gillespy2.TimeSpan.linspace(t=self.t_max, num_points=int(self.t_max*self.sampling))
        self.timespan(tspan)
        self.init_rxn()
        for stim in stim_vec:
            self.update_rxn(stim)
            results = self.run(number_of_trajectories=1,algorithm="Tau-Hybrid")
            tot_results.append(results)
            self.delete_all_events()
            updates = {}
            for name in self.get_all_species().keys():
                updates[name] = results[name][-1]
            self.get_updates(updates)
        return tot_results
    
    def run_online_rxn(self,updates,stim_vec):
        tspan = gillespy2.TimeSpan.linspace(t=self.t_max, num_points=int(self.t_max*self.sampling))
        self.timespan(tspan)
        if updates is not None:
            self.get_updates(updates)
        self.update_rxn(stim_vec)
        results = self.run(number_of_trajectories = 1, algorithm= 'Tau-Hybrid')
        new_state = self.give_updates(results)
        return results,new_state