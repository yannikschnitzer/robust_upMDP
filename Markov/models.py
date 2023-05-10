import numpy as np
import Markov.storm_interface as storm_ui

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

    def fix_pol(self, pol):
        """
        Fixes policy, returns an MC
        """
        fixed_MC = MC()
        fixed_MC.States = self.States
        fixed_MC.Init_state = self.Init_state
        trans_probs = []
        trans_ids = []
        for s in self.States:
            if len(self.Enabled_actions[s])>0:
                trans_probs_s = []
                trans_ids_s = []
                for act_num, act in enumerate(self.Enabled_actions[s]):
                    act_prob = pol[s][act]
                    trans_probs_s_a = [act_prob*p for p in self.Transition_probs[s][act_num]]
                    trans_ids_s_a = self.trans_ids[s][act_num]
                    for i, s_prime in enumerate(trans_ids_s_a):
                        if s_prime in trans_ids_s:
                            trans_probs_s[trans_ids_s.index(s_prime)] += trans_probs_s_a[i]
                        else:
                            trans_probs_s.append(trans_probs_s_a[i])
                            trans_ids_s.append(s_prime)
                trans_probs.append(trans_probs_s)
                trans_ids.append(trans_ids_s)
            else:
                trans_probs.append([])
                trans_ids.append([])
        fixed_MC.Transition_probs = trans_probs
        fixed_MC.trans_ids = trans_ids
        fixed_MC.Name = self.Name
        fixed_MC.Formulae = self.Formulae
        fixed_MC.Labels = self.Labels
        fixed_MC.Labelled_states = self.Labelled_states
        fixed_MC.opt = self.opt

        return fixed_MC

class pMDP(MDP):
    """
    Class for parametric MDPs
    Transition probabilities should now be functions over parameters
    """
    
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

class storm_upMDP:
    opt = "max"

    def sample_MDP(self):
        sample = storm_ui.sample_MDP(self.params, self.model, self.filename, self.weather)
        out = storm_MDP()
        out.mdp = sample
        out.props = self.props
        out.Init_state = self.Init_state
      
        out.Transition_probs = [[[transition.value() for transition in action.transitions] for action in state.actions] for state in sample.states]
        #out.trans_ids = [[[transition.column for transition in action.transitions] for action in state.actions] for state in sample.states]
        out.trans_ids = self.trans_ids
        out.States = self.States
        out.Actions = self.Actions
        out.Enabled_actions = [[int(str(action)) for action in state.actions] for state in sample.states]
        out.Labels = self.Labels
        out.Labelled_states = self.Labelled_states
        out.Formulae = self.Formulae
        out.opt = self.opt
        return out


class upMDP(pMDP):
    """
    upMDP with sampling functionality
    """

    param_sampler = None # Function to sample parameters

    def sample_MDP(self):
        params = self.param_sampler()
        return self.fix_params(params)
