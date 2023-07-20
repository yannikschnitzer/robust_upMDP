import Markov.models as Markov
import numpy as np
import Models.trans_funcs as t_f
import Models.samplers as samplers

def get_model():
    Hol_Model = Markov.upMDP()

    Hol_Model.States = np.array(range(5))
    # States : [0: Start, 1: Home, 2: Hol booked, 3: Happy, 4: Sad]
    Hol_Model.Actions = np.array(range(3))
    Hol_Model.Init_state = 0
    
    zero = t_f.fixed(0)
    one = t_f.fixed(1)
    
    funcs = [t_f.linear_multi(i) for i in range(8)]

    Hol_Model.Transition_probs = [
        [
            funcs[0],# Book Hol
            funcs[1] # Stay Home
        ],
        [
            funcs[2], # Hike
            funcs[3], # Gallery
            funcs[7]  # Late booking
        ],
        [
            funcs[4], # Hike
            funcs[5], # Gallery
            funcs[6]  # Cancel Hol
        ],
        [
            [one], # Happy
        ],
        [
            [one] # Sad
        ]]

    Hol_Model.trans_ids = [[[2, 4], [1, 4]],
                           [[3, 4], [3, 4], [2, 1]],
                           [[3, 4], [3, 4],[0,2]],
                           [[3]],
                           [[4]]]
    Hol_Model.paramed = [[[True, True],[True, True]],
                     [[True, True], [True, True], [True, True]],
                     [[True, True], [True, True], [True, True]], 
                     [[False]],
                     [[False]]]
    
    Hol_Model.max_supports = 7
    #Hol_Model.param_sampler = samplers.gauss(np.array([0.5]), np.array([[0.2]]))
    Hol_Model.param_sampler = samplers.uniform(8, [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], 
                                                  [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    Hol_Model.Labels = ["init", "happy"]
    Hol_Model.Labelled_states = [[0,1,2,3,4],[3]]

    Hol_Model.Name = "Holiday_model"
        
    Hol_Model.Formulae = ["Pmax=? [ F \"happy\"]"]

    Hol_Model.Enabled_actions = [[0,1],[0,1,2],[0,1,2],[0],[0]]

    return Hol_Model

