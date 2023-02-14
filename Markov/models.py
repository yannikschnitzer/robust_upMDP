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
                act = pol[s]
                trans_probs.append(self.Transition_probs[s][act])
                trans_ids.append(self.trans_ids[s][act])
            else:
                trans_probs.append([])
                trans_ids.append([])
        fixed_MC.Transition_probs = trans_probs
        fixed_MC.trans_ids = trans_ids
        fixed_MC.Name = self.Name
        fixed_MC.Formulae = self.Formulae
        fixed_MC.Labels = self.Labels
        fixed_MC.Labelled_states = self.Labelled_states

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

        return fixed_MDP

class storm_MDP:
    mdp = None
    props = None

class storm_upMDP:

    def sample_MDP(self):
        sample = storm_ui.sample_MDP(self.params, self.model, self.filename, self.weather)
        out = storm_MDP()
        out.mdp = sample
        out.props = self.props
        out.Init_state = self.Init_state
        return out


class upMDP(pMDP):
    """
    upMDP with sampling functionality
    """

    param_sampler = None # Function to sample parameters

    def sample_MDP(self):
        params = self.param_sampler()
        return self.fix_params(params)
