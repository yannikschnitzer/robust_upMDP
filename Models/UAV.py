import Markov.models as Markov
import numpy as np
import Models.trans_funcs as t_f
import Models.samplers as samplers

def get_model():
    Model = Markov.upMDP()

    Model.States = np.array(range(10000))
    
    states_per_dim = 22

    Model.Actions = np.array(range(6)) #N, S, E, W, up, down
    Model.Init_state = 0
    
    Model.trans_ids = [[    
                             [s_prime, 
                              s_prime-1, 
                              s_prime+1,
                              s_prime+states_per_dim, 
                              s_prime-states_per_dim, 
                              s_prime+states_per_dim**2, 
                              s_prime-states_per_dim**2]
                            for s_prime in [s+1, s-1, s+states_per_dim, s-states_per_dim, s+states_per_dim**2, s-states_per_dim**2]
                            ]
                            for s in Model.States]

    zero = t_f.fixed(0)
    one = t_f.fixed(1)
    
    Model.Transition_probs = [
        [
            [one],
            [one], # from init
            [one],
        ],
        [
            [t_f.linear, t_f.one_minus_linear],
        ],
        [
            [t_f.fixed(0.51), t_f.fixed(0.49)],
        ],
        [
            [t_f.one_minus_linear, t_f.linear],
        ],
        [
        ],
        [
        ]]

    Model.param_sampler = samplers.gauss(0.5, 0.2)
    
    Model.Labels = ["init", "reached"]
    Model.Labelled_states = [list(range(1000)),[989]]

    Model.Name = "UAV"
    
    Model.Formulae = ["Pmax=? [ F \"reached\"]"]

    Model.Enabled_actions = [list(range(6)) for i in Model.States]

    return Model
