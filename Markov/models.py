from itertools import chain
import numpy as np
import Markov.storm_interface as storm_ui
import core.create_drone_prism_model as create_prism
import stormpy
import stormpy.core
import stormpy.logic
import stormpy.pars
import stormpy.examples
import stormpy.examples.files
from time import perf_counter as timer
from multiprocessing import Pool

class base:
    """
    Base class for Markov Models
    """
    States = None
    Actions = None
    Transition_probs = None
    Init_state = None
    Labels = None
    Labelled_states = None
    Name = None
    trans_ids = None
    Formulae = None
    opt = "max"
    Enabled_actions = None

    def __init__(self, model=None):
        if model is not None:
            self.States = model.States
            self.Actions = model.Actions
            self.Transition_probs = model.Transition_probs
            self.Init_state = model.Init_state
            self.Labels = model.Labels
            self.Labelled_states = model.Labelled_states
            self.Name = model.Name
            self.trans_ids = model.trans_ids
            self.Formulae = model.Formulae
            self.opt = model.opt
            self.Enabled_actions = model.Enabled_actions

class MC(base):
    """
    Markov Chain
    """
    pass

class MDP(base):
    """
    Class for standard MDPs
    """
    
    def fix_state_pol(self, pol, s):
        num_enabled = len(self.Enabled_actions[s])
        if num_enabled > 1:
            num_enabled = len(self.Enabled_actions[s])
            trans_arr = np.zeros((num_enabled,len(self.States)))
            for i, inds in enumerate(self.trans_ids[s]):
                trans_arr[i][inds] = self.Transition_probs[s][i]
            res = pol[self.Enabled_actions[s]]@trans_arr
            s_primes = list(np.argwhere(res).flatten())
            s_trans_probs = list(res[s_primes])
        else:
            s_primes = list(chain(*self.trans_ids[s]))
            s_trans_probs = self.Transition_probs[s][0]
        return s_primes, s_trans_probs
    
    def fix_pol(self, pol):
        new_trans_probs = []
        new_trans_ids = []
        parallel = True
        for s in self.States:
            s_primes, s_probs = self.fix_state_pol(pol[s],s)
            new_trans_ids.append(s_primes)
            new_trans_probs.append(s_probs)
        fixed_MC = MC()
        fixed_MC.States = self.States
        fixed_MC.Init_state = self.Init_state
        fixed_MC.Transition_probs = new_trans_probs
        fixed_MC.trans_ids = new_trans_ids
        fixed_MC.Name = self.Name
        fixed_MC.Formulae = self.Formulae
        fixed_MC.Labels = self.Labels
        fixed_MC.Labelled_states = self.Labelled_states
        fixed_MC.opt = self.opt
        return fixed_MC

    #def fix_pol(self, pol):
    #    """
    #    Fixes policy, returns an MC
    #    """
    #    fixed_MC = MC()
    #    fixed_MC.States = self.States
    #    fixed_MC.Init_state = self.Init_state
    #    trans_probs = []
    #    trans_ids = []
    #    for s in self.States:
    #        if len(self.Enabled_actions[s])>0:
    #            trans_probs_s = []
    #            trans_ids_s = []
    #            for act_num, act in enumerate(self.Enabled_actions[s]):
    #                act_prob = pol[s][act]
    #                trans_probs_s_a = [act_prob*p for p in self.Transition_probs[s][act_num]]
    #                trans_ids_s_a = self.trans_ids[s][act_num]
    #                for i, s_prime in enumerate(trans_ids_s_a):
    #                    if s_prime in trans_ids_s:
    #                        trans_probs_s[trans_ids_s.index(s_prime)] += trans_probs_s_a[i]
    #                    else:
    #                        trans_probs_s.append(trans_probs_s_a[i])
    #                        trans_ids_s.append(s_prime)
    #            trans_probs.append(trans_probs_s)
    #            trans_ids.append(trans_ids_s)
    #        else:
    #            trans_probs.append([])
    #            trans_ids.append([])
    #    fixed_MC.Transition_probs = trans_probs
    #    fixed_MC.trans_ids = trans_ids
    #    fixed_MC.Name = self.Name
    #    fixed_MC.Formulae = self.Formulae
    #    fixed_MC.Labels = self.Labels
    #    fixed_MC.Labelled_states = self.Labelled_states
    #    fixed_MC.opt = self.opt

    #    return fixed_MC

class iMDP(MDP):
    Transition_probs = None

class pMDP(MDP):
    """
    Class for parametric MDPs
    Transition probabilities should now be functions over parameters
    """
    def build_imdp(self, params):
        fixed_iMDP = iMDP()
        fixed_iMDP.States = self.States
        fixed_iMDP.Actions = self.Actions
        fixed_iMDP.Init_state = self.Init_state
        trans_probs = []
        for s in self.States:
            trans_probs_s = []
            for a_id, a in enumerate(self.Enabled_actions[s]):
                trans_probs_s_a = []
                for s_prime_id, s_prime in enumerate(self.trans_ids[s][a]):
                    min_p = 1
                    max_p = 0
                    for param in params:
                        new_p = self.Transition_probs[s][a_id][s_prime_id](param)
                        min_p = min(min_p, new_p)
                        max_p = max(max_p, new_p)
                    trans_probs_s_a.append((min_p, max_p))
                trans_probs_s.append(trans_probs_s_a)
            trans_probs.append(trans_probs_s)
        fixed_iMDP.Transition_probs = trans_probs
        fixed_iMDP.supports = set()
        for s in self.States:
            for a_id, a in enumerate(self.Enabled_actions[s]):
                for s_prime_id, s_prime in enumerate(self.trans_ids[s][a]):
                    if self.paramed[s][a_id][s_prime_id]:
                        for i, param in enumerate(params):
                            p = self.Transition_probs[s][a_id][s_prime_id](param)
                            if p in trans_probs[s][a_id][s_prime_id]: 
                                fixed_iMDP.supports.add(i) # if sample defines ub OR lb, might be support
        fixed_iMDP.Labels = self.Labels
        fixed_iMDP.Labelled_states = self.Labelled_states
        fixed_iMDP.Name = self.Name
        fixed_iMDP.Formulae = []
        for formula in self.Formulae:
            if "Pmax" in formula:
                fixed_iMDP.Formulae.append("Pmaxmin"+formula[4:])
            elif "Pmin" in formula:
                fixed_iMDP.Formulae.append("Pminmax"+formula[4:])
        fixed_iMDP.Formulae = self.Formulae
        fixed_iMDP.Enabled_actions = self.Enabled_actions
        fixed_iMDP.trans_ids = self.trans_ids
        fixed_iMDP.opt = self.opt
        fixed_iMDP.params = params

        return fixed_iMDP



    def fix_params(self, params):
        fixed_MDP = MDP()
        fixed_MDP.States = self.States
        fixed_MDP.Actions = self.Actions
        fixed_MDP.Init_state = self.Init_state
        trans_probs = []
        for s in self.States:
            trans_probs_s = []
            for a in self.Enabled_actions[s]:
                trans_probs_s_a = []
                for s_prime_id, s_prime in enumerate(self.trans_ids[s][a]):
                    trans_probs_s_a.append(self.Transition_probs[s][a][s_prime_id](params))
                trans_probs_s.append(trans_probs_s_a)
            trans_probs.append(trans_probs_s)
        fixed_MDP.Transition_probs = trans_probs
        fixed_MDP.Labels = self.Labels
        fixed_MDP.Labelled_states = self.Labelled_states
        fixed_MDP.Name = self.Name
        fixed_MDP.Formulae = self.Formulae
        fixed_MDP.Enabled_actions = self.Enabled_actions
        fixed_MDP.trans_ids = self.trans_ids
        fixed_MDP.opt = self.opt
        fixed_MDP.params = params

        return fixed_MDP


class storm_MDP:
    mdp = None
    props = None
    
    States = None
    Actions = None
    Transition_probs = None
    Init_state = None
    Labels = None
    Labelled_states = None
    Name = None
    trans_ids = None
    Formulae = None
    opt = "max"

    def fix_state_pol(self, pol, s):
        if len(self.mdp.states[s].actions) > 1:
            num_enabled = len(self.Enabled_actions[s])
            trans_arr = np.zeros((num_enabled,len(self.States)))
            for i, inds in enumerate(self.trans_ids[s]):
                trans_arr[i][inds] = self.Transition_probs[s][i]
            res = pol[self.Enabled_actions[s]]@trans_arr
            s_primes = list(np.argwhere(res).flatten())
            s_trans_probs = list(res[s_primes])
        else:
            s_primes = list(chain(*self.trans_ids[s]))
            s_trans_probs = self.Transition_probs[s][0]
        return s_primes, s_trans_probs


    def fix_pol(self, pol):
        new_trans_probs = []
        new_trans_ids = []
        parallel = True
        for s in self.States:
            s_primes, s_probs = self.fix_state_pol(pol[s],s)
            new_trans_ids.append(s_primes)
            new_trans_probs.append(s_probs)
        fixed_MC = MC()
        fixed_MC.States = self.States
        fixed_MC.Init_state = self.Init_state
        fixed_MC.Transition_probs = new_trans_probs
        fixed_MC.trans_ids = new_trans_ids
        fixed_MC.Name = self.Name
        fixed_MC.Formulae = self.Formulae
        fixed_MC.Labels = self.Labels
        fixed_MC.Labelled_states = self.Labelled_states
        fixed_MC.opt = self.opt
        return fixed_MC

class storm_upMDP:
    opt = "max"
    
    def param_sampler(self):
    
        if 'drone' in self.filename:
            drone = True
        else:
            drone = False
        
        # If we are performing the drone benchmark
        if drone:
            groups = create_prism.parameter_groups()
            
            # Instantiate parameters in the model
            create_prism.parameter_definitions(groups,False)
                
        # Sample parameters according to the region defined by
        # "Parameter synthesis for Markov models: Faster than ever" paper
        point=dict()

        # Switch between drone and other benchmarks
        if drone:
            param_inst = create_prism.parameter_dirichlet_instantiations(
                groups, self.weather)
            param_samps = create_prism.parameter_dirichlet_samples(param_inst)
            
            for x in self.params:

            #instantiate parameters
                parameter_group = storm_ui.search(param_inst, str(x.name))

                element_int=0
                for element in parameter_group:
                    
                    param_sample_array = param_samps[tuple(parameter_group)]
                    if (str(element)) == (str(x.name)):

                        point[x] = param_sample_array[0,element_int]
                    element_int=element_int+1
        else:
            
            for x in self.params:
                if "coin" in self.filename:
                    s = np.random.uniform(0.2, 0.8)
        
                else:
                    s = np.random.uniform(1e-5,1-1e-5)
        
                point[x] = stormpy.RationalRF(s)
        
        # Assign parameter values to model
        rational_parameter_assignments = dict(
            [[x, stormpy.RationalRF(val)] for x, val in point.items()])
        
        # Instantiate model
            
        return rational_parameter_assignments

    def fix_params(self, params):
        start_time = timer()
        times = []
        if str(self.model.model_type)=='ModelType.DTMC':
            instantiator = stormpy.pars.PDtmcInstantiator(self.model)
        elif  str(self.model.model_type)=='ModelType.MDP':
            instantiator = stormpy.pars.PMdpInstantiator(self.model)
        else:
            raise RuntimeError("Invalid model type (should be a pDTMC or pMDP).")

        sample = instantiator.instantiate(params)
        out = storm_MDP()
        out.mdp = sample
        out.props = self.props
        out.Init_state = self.Init_state

        times.append(timer()-start_time)
        out.Transition_probs = self.Transition_probs
        states = sample.states
        for s in self.paramed_states:
            state = states[s]
            out.Transition_probs[s] =   [
                                            [   
                                                transition.value() for transition in action.transitions
                                            ] 
                                            for action in state.actions
                                        ] 
        #out.trans_ids = [[[transition.column for transition in action.transitions] for action in state.actions] for state in sample.states]
        
        times.append(timer()-start_time)
        out.trans_ids = self.trans_ids
        out.States = self.States
        out.Actions = self.Actions
        out.Enabled_actions = self.Enabled_actions
        #out.Enabled_actions = [[int(str(action)) for action in state.actions] for state in sample.states]
        times.append(timer()-start_time)
        out.Labels = self.Labels
        out.Labelled_states = self.Labelled_states
        out.Formulae = self.Formulae
        out.opt = self.opt
        out.params = params
        out.Name = self.Name
        #print(times)
        return out
    

    def sample_MDP(self):
        acc_params = self.param_sampler()
        sample = self.fix_params(acc_params)
        return sample

    def get_trans_arr(self):
        arr  = np.zeros((len(self.States),len(self.Actions),len(self.States)))
        for s_id, s in enumerate(self.States):
            for a_id, a in enumerate(self.Enabled_actions[s]):
                for s_prime_id, s_prime in enumerate(self.trans_ids[s][a]):
                    trans_prob = self.Transition_probs[s_id][a_id][s_prime_id]
                    if trans_prob != 'p':
                        arr[s][a][s_prime] = trans_prob
        return arr


class upMDP(pMDP):
    """
    upMDP with sampling functionality
    """

    param_sampler = None # Function to sample parameters

    def sample_MDP(self):
        params = self.param_sampler()
        return self.fix_params(params)
    
    def get_trans_arr(self):
        arr  = np.zeros((len(self.States),len(self.Actions),len(self.States)))
        for s_id, s in enumerate(self.States):
            for a_id, a in enumerate(self.Enabled_actions[s]):
                for s_prime_id, s_prime in enumerate(self.trans_ids[s][a]):
                    if not self.paramed[s_id][a_id][s_prime_id]:
                        trans_prob = self.Transition_probs[s_id][a_id][s_prime_id](0)
                        arr[s][a][s_prime] = trans_prob
        return arr
