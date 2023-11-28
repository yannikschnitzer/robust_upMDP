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
import multiprocessing as mp
import copy

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
    gamma = 0.99
    rho = None
    switch_res = False

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
            self.gamma = model.gamma
            self.rho = model.rho

    def check_timeout(self, start):
        if timer() - start > self.max_time:
            print("Timed out!")
            return True
        else:
            return False
    
    def make_max(self):
        new_model = copy.copy(self)
        if self.opt == "min":
            new_model.opt ="max"
            new_formulae = []
            for spec in self.Formulae:
                spec_loc = spec.index("min")
                new_spec = list(spec)
                new_spec[spec_loc:spec_loc+3] = "max"
               
                if "F" in spec:
                    F_loc = spec.index("F")
                    new_spec[F_loc] = "G"
                
                if "&" in spec:
                    and_loc = spec.index("&")
                    new_spec[and_loc] = "|"
                
                if "|" in spec:
                    or_loc = spec.index("|")
                    new_spec[or_loc] = "&"
                
                add = True
                idxs = []
                counter = 0
                for elem in spec:
                    counter += 1
                    if elem == '"':
                        if add:
                            idxs.append(counter-1)
                        add = not add
                for count, i in enumerate(idxs):
                    new_spec.insert(i+count, "!")
                new_formulae.append(''.join(new_spec))
            new_model.Formulae = new_formulae
            new_model.switch_res = True
        return new_model

class MC(base):
    """
    Markov Chain
    """
    pass

class MDP(base):
    """
    Class for standard MDPs
    """
    
    parallel = False # Setting to true leads to slower performance for all models

    def fix_state_pol(self, args):
        pol_all = args[0]
        s_all = args[1]
        if type(s_all) is not list:
            s_all = [s_all]
            pol_all = [pol_all]
        s_p_all = []
        s_t_all = []
        for s, pol in zip(s_all, pol_all):
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
            s_p_all.append(s_primes)
            s_t_all.append(s_trans_probs)
        return s_p_all, s_t_all
    
    def fix_pol(self, pol):
        
        num_batches = mp.cpu_count() 
        
        batch_size = len(self.States)//num_batches+1
        
        # parallelisation slows things down here...
        #acts_bool = [[True  if a in model.Enabled_actions[s] else False for a in model.Actions] for s in model.States]
        pol_list = [pol[s] for s in self.States]

        if len(self.States) <= 320:
            args = zip(pol_list, self.States)
            
            if not self.parallel:
                res = [self.fix_state_pol(arg) for arg in args]
            else:
                with mp.Pool() as p:
                    res = p.map(self.fix_state_pol, args)
        else:
            args_batched = [(pol_list[x:x+batch_size],self.States[x:x+batch_size]) for x in range(0,len(pol_list),batch_size) ]
            if not self.parallel:
                res = [self.fix_state_pol(arg) for arg in args_batched]
            else:
                with mp.Pool() as p:
                    res = p.map(self.fix_state_pol, args_batched)#, chunksize=batch_size)
        
        new_trans_probs = []
        new_trans_ids = []
        for elem in res:
            new_trans_ids += elem[0] 
            new_trans_probs += elem[1] 
        #import pdb; pdb.set_trace()
        #for s in self.States:
        #    s_primes, s_probs = self.fix_state_pol(pol[s],s)
        #    new_trans_ids.append(s_primes)
        #    new_trans_probs.append(s_probs)
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
        fixed_MC.gamma = self.gamma
        fixed_MC.rho = self.rho
        return fixed_MC

class iMDP(MDP):
    Transition_probs = None

class pMDP(MDP):
    """
    Class for parametric MDPs
    Transition probabilities should now be functions over parameters
    """
    def build_imdp(self, params):
        start = timer()
        fixed_iMDP = iMDP()
        fixed_iMDP.States = self.States
        fixed_iMDP.Actions = self.Actions
        fixed_iMDP.Init_state = self.Init_state
        trans_probs = []
        
        fixed_iMDP.supports = set()
        
        for s in self.States:
            if self.check_timeout(start):
                return None
            trans_probs_s = []
            for a_id, a in enumerate(self.Enabled_actions[s]):
                trans_probs_s_a = []
                for s_prime_id, s_prime in enumerate(self.trans_ids[s][a_id]):
                    min_p = 1
                    max_p = 0
                    supps = [-1, -1]
                    for i, param in enumerate(params):
                        new_p = float(self.Transition_probs[s][a_id][s_prime_id](param))
                        if new_p < min_p:
                            supps[0] = i
                            min_p = new_p
                        elif new_p > max_p:
                            supps[1] = i
                            max_p = new_p
                    trans_probs_s_a.append((min_p, max_p))
                    if self.paramed[s][a_id][s_prime_id]:
                        fixed_iMDP.supports.add(supps[0])
                        fixed_iMDP.supports.add(supps[1])
                trans_probs_s.append(trans_probs_s_a)
            trans_probs.append(trans_probs_s)
        fixed_iMDP.Transition_probs = trans_probs
        #for s in self.States:`
        #    if self.check_timeout(start):
        #        return None
        #    for a_id, a in enumerate(self.Enabled_actions[s]):
        #        for s_prime_id, s_prime in enumerate(self.trans_ids[s][a_id]):
        #            if self.paramed[s][a_id][s_prime_id]:
        #                for i, param in enumerate(params):
        #                    p = self.Transition_probs[s][a_id][s_prime_id](param)
        #                    if p in trans_probs[s][a_id][s_prime_id]: 
        #                        fixed_iMDP.supports.add(i) # if sample defines ub OR lb, might be support
        fixed_iMDP.Labels = self.Labels
        fixed_iMDP.Labelled_states = self.Labelled_states
        fixed_iMDP.Name = self.Name
        fixed_iMDP.Formulae = []
        for formula in self.Formulae:
            if "Pmax" in formula:
                fixed_iMDP.Formulae.append("Pmaxmin"+formula[4:])
            elif "Pmin" in formula:
                fixed_iMDP.Formulae.append("Pminmax"+formula[4:])
        #import pdb; pdb.set_trace()
        #fixed_iMDP.Formulae = self.Formulae
        fixed_iMDP.Enabled_actions = self.Enabled_actions
        fixed_iMDP.trans_ids = self.trans_ids
        fixed_iMDP.opt = self.opt
        fixed_iMDP.params = params
        
        fixed_iMDP.gamma = self.gamma
        fixed_iMDP.rho = self.rho

        return fixed_iMDP



    def fix_params(self, params):
        fixed_MDP = MDP()
        fixed_MDP.States = self.States
        fixed_MDP.Actions = self.Actions
        fixed_MDP.Init_state = self.Init_state
        trans_probs = []
        for s in self.States:
            trans_probs_s = []
            for a_id, a in enumerate(self.Enabled_actions[s]):
                trans_probs_s_a = []
                for s_prime_id, s_prime in enumerate(self.trans_ids[s][a_id]):
                    trans_probs_s_a.append(self.Transition_probs[s][a_id][s_prime_id](params))
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
        fixed_MDP.gamma = self.gamma
        fixed_MDP.rho = self.rho

        return fixed_MDP


class storm_MDP(MDP):
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


class storm_upMDP(base):
    opt = "max"
    
    def parameter_dist(self):
        if "coin" in self.filename:
            s = np.random.uniform(0.2, 0.8)
        
        else:
            #s = np.random.uniform(1e-5, 1-1e-5) # This was Thom's distribution
            s = (np.pi+np.random.vonmises(0,5))/(2*np.pi) # This is a more interesting distribution
        return s


    def build_imdp(self, params):
        if str(self.model.model_type)=='ModelType.DTMC':
            instantiator = stormpy.pars.PDtmcInstantiator(self.model)
        elif  str(self.model.model_type)=='ModelType.MDP':
            instantiator = stormpy.pars.PMdpInstantiator(self.model)
        else:
            raise RuntimeError("Invalid model type (should be a pDTMC or pMDP).")
        fixed_iMDP = iMDP()
        fixed_iMDP.States = self.States
        fixed_iMDP.Actions = self.Actions
        fixed_iMDP.Init_state = self.Init_state
        Transition_probs = copy.copy(self.Transition_probs)
        start = True
        for param in params:
            sample = instantiator.instantiate(param)
            states = sample.states
            if start:
                state_set = self.States
                start = False
            else:
                state_set = self.paramed_states
            for s in state_set:
                state = states[s]
                Transition_probs[s] =   [
                                                [   
                                                    (transition.value(),transition.value()) 
                                                    if type(Transition_probs[s][a_id][s_prime_id]) is not tuple 
                                                    else
                                                    (min(Transition_probs[s][a_id][s_prime_id][0], transition.value()),max(Transition_probs[s][a_id][s_prime_id][1], transition.value()))
                                                    for s_prime_id, transition in enumerate(action.transitions) 
                                                ] 
                                                for a_id, action in enumerate(state.actions)
                                            ] 
        
        fixed_iMDP.Transition_probs = Transition_probs
        fixed_iMDP.supports = set()
        for i, param in enumerate(params):
            sample = instantiator.instantiate(param)
            states = sample.states
            for s in self.paramed_states:
                state = states[s]
                for a_id, action in enumerate(state.actions):
                    for s_prime_id, transition in enumerate(action.transitions):
                        if transition.value() in Transition_probs[s][a_id][s_prime_id]:
                            fixed_iMDP.supports.add(i)
        fixed_iMDP.Labels = self.Labels
        fixed_iMDP.Labelled_states = self.Labelled_states
        fixed_iMDP.Name = self.Name
        fixed_iMDP.Formulae = []
        for formula in self.Formulae:
            if "Pmax" in formula:
                fixed_iMDP.Formulae.append("Pmaxmin"+formula[4:])
            elif "Pmin" in formula:
                fixed_iMDP.Formulae.append("Pminmax"+formula[4:])
        #fixed_iMDP.Formulae = self.Formulae
        fixed_iMDP.Enabled_actions = self.Enabled_actions
        fixed_iMDP.trans_ids = self.trans_ids
        fixed_iMDP.opt = self.opt
        fixed_iMDP.params = params
        fixed_iMDP.gamma = self.gamma
        fixed_iMDP.rho = self.rho

        return fixed_iMDP
    
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
                s = self.parameter_dist()
                #if "coin" in self.filename:
                #    s = np.random.uniform(0.2, 0.8)
        
                #else:
                #    s = np.random.uniform(1e-5,1-1e-5)

        
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
        out.gamma = self.gamma
        out.rho = self.rho
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
                for s_prime_id, s_prime in enumerate(self.trans_ids[s][a_id]):
                    if not self.paramed[s_id][a_id][s_prime_id]:
                        trans_prob = self.Transition_probs[s_id][a_id][s_prime_id](0)
                        arr[s][a][s_prime] = trans_prob
        return arr
