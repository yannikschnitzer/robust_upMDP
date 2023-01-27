import Markov.models as Markov
import numpy as np
import Models.trans_funcs as t_f
import Models.samplers as samplers

def get_model():
    Test_Model = Markov.upMDP()

    Test_Model.States = np.array(range(6))
    Test_Model.Actions = np.array(range(3))
    Test_Model.Init_state = 0
    
    zero = t_f.fixed(0)
    one = t_f.fixed(1)
   
    test_val = 0.49

    Test_Model.Transition_probs = [
        [
            [one],
            [one], # from init
            [one],
        ],
        [
            [t_f.linear, t_f.one_minus_linear],
        ],
        [
            [t_f.fixed(test_val), t_f.fixed(1-test_val)],
        ],
        [
            [t_f.one_minus_linear, t_f.linear],
        ],
        [
            [one]
        ],
        [
            [one]
        ]]

    Test_Model.trans_ids = [[[1],[2],[3]],[[4,5]],[[4,5]],[[4,5]],[[4]],[[5]]]

    Test_Model.param_sampler = samplers.gauss(0.5, 0.2)
    
    Test_Model.Labels = ["init", "reached"]
    Test_Model.Labelled_states = [[0,1,2,3,4,5],[4]]

    Test_Model.Name = "test"
        
    Test_Model.Formulae = ["Pmax=? [ F \"reached\"]"]

    Test_Model.Enabled_actions = [[0,1,2],[0],[0],[0],[0],[0]]

    return Test_Model
