import Markov.models as Markov
import numpy as np
import Models.trans_funcs as t_f
import Models.samplers as samplers

def get_model():
    Test_Model = Markov.upMDP()

    Test_Model.States = np.array(range(6))
    Test_Model.Actions = np.array(range(3))
    Test_Model.Init_state = 0
    
    zero = t_f.fixed(0.0)
    one = t_f.fixed(1.0)
   
    test_val = 0.52

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
    Test_Model.paramed = [[[False],[False],[False]],
                     [[True, True]],
                     [[False, False]], 
                     [[True,True]], 
                     [[False]],
                     [[False]]]
    Test_Model.max_supports = 3
    Test_Model.param_sampler = samplers.gauss(np.array([0.5]), np.array([[0.2]]))
    
    Test_Model.Labels = ["init", "reached"]
    Test_Model.Labelled_states = [[0,1,2,3,4,5],[4]]

    Test_Model.Name = "test"
        
    Test_Model.Formulae = ["Pmax=? [ F \"reached\"]"]

    Test_Model.Enabled_actions = [[0,1,2],[0],[0],[0],[0],[0]]

    return Test_Model

def get_model_2():
    Test_Model = Markov.upMDP()

    Test_Model.States = np.array(range(5))
    Test_Model.Actions = np.array(range(2))
    Test_Model.Init_state = 0
    
    zero = t_f.fixed(0.0)
    one = t_f.fixed(1.0)
    
    funcs = [t_f.linear_multi(i) for i in range(2)]


    Test_Model.Transition_probs = [
        [
            [one],
            [one],
        ],
        [
            funcs[0],
        ],
        [
            funcs[1],
        ],
        [
            [one]
        ],
        [
            [one]
        ]]

    Test_Model.paramed = [[[False],[False]],
                     [[True, True]],
                     [[True, True]], 
                     [[False]],
                     [[False]]]
    Test_Model.max_supports = 2
    Test_Model.trans_ids = [[[1],[2]],[[3,4]],[[3,4]],[[3]],[[4]]]

    #Test_Model.param_sampler = samplers.gauss(np.ones((2))*0.8, np.eye(2)*0.1)
    Test_Model.param_sampler = samplers.uniform(2, [0.2, 0.2], [0.5,0.5])

    Test_Model.Labels = ["init", "reached"]
    Test_Model.Labelled_states = [[0,1,2,3,4],[3]]
    Test_Model.Labels = ["init", "reached"]
    Test_Model.Labelled_states = [[0],[3]]

    Test_Model.Name = "test"
        
    Test_Model.Formulae = ["Pmax=? [ F \"reached\"]"]
    
    Test_Model.Formulae = ["Pmin=? [ G !\"reached\"]"]
    Test_Model.opt = "min"

    Test_Model.Enabled_actions = [[0,1],[0],[0],[0],[0]]

    return Test_Model

def get_model_3():
    Test_Model = Markov.upMDP()

    Test_Model.States = np.array(range(5))
    Test_Model.Actions = np.array(range(2))
    Test_Model.Init_state = 0
    
    zero = t_f.fixed(0.0)
    one = t_f.fixed(1.0)
    
    funcs = [t_f.linear_multi(i) for i in range(2)]


    Test_Model.Transition_probs = [
        [
            [one],
        ],
        [
            [zero,one],
            funcs[0],
        ],
        [
            [zero,one],
            funcs[1],
        ],
        [
            [one]
        ],
        [
            [one]
        ]]
    Test_Model.paramed = [[[False]],
                     [[False, False],[True, True]],
                     [[False, False],[True,True]], 
                     [[False]],
                     [[False]]]

    Test_Model.trans_ids = [[[1]],[[2,4],[2,4]],[[3,4],[3,4]],[[3]],[[4]]]

    Test_Model.param_sampler = samplers.gauss(np.ones((2))*0.85, np.eye(2)*0.1)
    Test_Model.max_supports = 2
    
    Test_Model.Labels = ["init", "reached"]
    Test_Model.Labelled_states = [[0,1,2,3,4],[3]]

    Test_Model.Name = "test"
        
    Test_Model.Formulae = ["Pmax=? [ F \"reached\"]"]

    Test_Model.Enabled_actions = [[0],[0,1],[0,1],[0],[0]]

    return Test_Model

def get_model_4():
    Test_Model = Markov.upMDP()

    Test_Model.States = np.array(range(5))
    Test_Model.Actions = np.array(range(2))
    Test_Model.Init_state = 0
    
    zero = t_f.fixed(0)
    one = t_f.fixed(1)
    
    funcs = [t_f.linear_multi(i) for i in range(3)]


    Test_Model.Transition_probs = [
        [
            [one],
        ],
        [
            funcs[0],
        ],
        [
            funcs[1],
            funcs[2],
        ],
        [
            [one]
        ],
        [
            [one]
        ]]

    Test_Model.trans_ids = [[[1]],[[2,4]],[[3,4],[3,4]],[[3]],[[4]]]
    Test_Model.paramed = [[[False]],
                     [[True, True]],
                     [[True,True],[True,True]], 
                     [[False]],
                     [[False]]]

    #Test_Model.param_sampler = samplers.gauss(np.ones((2))*0.8, np.eye(2)*0.1)
    #Test_Model.param_sampler = samplers.uniform(2, [0.01, 0.2], [0.5,0.6])
    Test_Model.param_sampler = samplers.uniform(3, [0.8, 0.2, 0.9], [0.9,0.6, 0.95])
    Test_Model.max_supports = 2
    Test_Model.Labels = ["init", "reached"]
    Test_Model.Labelled_states = [[0,1,2,3,4],[3]]

    Test_Model.Name = "test"
        
    Test_Model.Formulae = ["Pmax=? [ F \"reached\"]"]

    Test_Model.Enabled_actions = [[0],[0],[0,1],[0],[0]]

    return Test_Model

def get_model_5():
    Test_Model = Markov.upMDP()

    Test_Model.States = np.array(range(9))
    Test_Model.Actions = np.array(range(2))
    Test_Model.Init_state = 0
    
    zero = t_f.fixed(0)
    one = t_f.fixed(1)
    
    funcs = [t_f.linear_multi(i) for i in range(4)]


    Test_Model.Transition_probs = [
        [
            [one],
            [one],
        ],
        [
            [one],
            [one],
        ],
        [
            funcs[0],
        ],
        [
            funcs[1],
        ],
        [
            [one],
            [one],
        ],
        [
            funcs[2],
        ],
        [
            funcs[3],
        ],
        [
            [one]
        ],
        [
            [one]
        ]]

    Test_Model.paramed = [[[False],[False]],
                     [[False], [False]],
                     [[True, True]], 
                     [[True, True]],
                     [[False], [False]],
                     [[True, True]], 
                     [[True, True]], 
                     [[False]],
                     [[False]]]
    Test_Model.max_supports = 4
    Test_Model.trans_ids = [[[1],[4]],[[2],[3]],[[7,8]],[[7,8]],[[5],[6]],[[7,8]],[[7,8]],[[7]],[[8]]]

    #Test_Model.param_sampler = samplers.gauss(np.ones((2))*0.8, np.eye(2)*0.1)
    Test_Model.param_sampler = samplers.uniform(4, [0.2, 0.4, 0.3, 0.1], [0.3,0.5,0.5,0.9])

    Test_Model.Labels = ["init", "reached"]
    Test_Model.Labelled_states = [[0,1,2,3,4,5,6,7,8],[7]]

    Test_Model.Name = "test"
        
    Test_Model.Formulae = ["Pmax=? [ F \"reached\"]"]

    Test_Model.Enabled_actions = [[0,1],[0,1],[0],[0],[0,1],[0],[0],[0],[0]]

    return Test_Model

def get_model_6():
    Test_Model = Markov.upMDP()

    Test_Model.States = np.array(range(3))
    Test_Model.Actions = np.array(range(6))
    Test_Model.Init_state = 0
    
    zero = t_f.fixed(0)
    one = t_f.fixed(1)
    
    funcs = [t_f.linear_multi(i) for i in range(6)]


    Test_Model.Transition_probs = [
            funcs,
        [
            [one]
        ],
        [
            [one]
        ]]

    Test_Model.paramed = [[[True, True],[True, True],[True, True],[True, True],[True, True],[True, True]],
                     [[False]],
                     [[False]]]
    Test_Model.max_supports = 6
    Test_Model.trans_ids = [[[1,2],[1,2],[1,2],[1,2],[1,2],[1,2]],[[1]],[[2]]]

    #Test_Model.param_sampler = samplers.gauss(np.ones((2))*0.8, np.eye(2)*0.1)
    Test_Model.param_sampler = samplers.uniform(6, 0.2,0.8)

    Test_Model.Labels = ["init", "reached"]
    Test_Model.Labelled_states = [[0,1,2],[1]]

    Test_Model.Name = "test"
        
    Test_Model.Formulae = ["Pmax=? [ F \"reached\"]"]

    Test_Model.Enabled_actions = [[0,1,2,3,4,5,],[0],[0]]

    return Test_Model

def get_model_7():
    Test_Model = Markov.upMDP()

    Test_Model.States = np.array(range(9))
    Test_Model.Actions = np.array(range(2))
    Test_Model.Init_state = 0
    
    zero = t_f.fixed(0)
    one = t_f.fixed(1)
    
    funcs = [t_f.linear_multi(i) for i in range(8)]


    Test_Model.Transition_probs = [
        [
            [one],
            [one],
        ],
        [
            funcs[0],
            funcs[1],
        ],
        [
            funcs[2],
        ],
        [
            funcs[3],
        ],
        [
            funcs[4],
            funcs[5],
        ],
        [
            funcs[6],
        ],
        [
            funcs[7],
        ],
        [
            [one]
        ],
        [
            [one]
        ]]

    Test_Model.paramed = [[[False],[False]],
                     [[True, True], [True, True]],
                     [[True, True]], 
                     [[True, True]],
                     [[True, True], [True, True]],
                     [[True, True]], 
                     [[True, True]], 
                     [[False]],
                     [[False]]]
    Test_Model.max_supports = 8
    Test_Model.trans_ids = [[[1],[4]],[[2,8],[3,8]],[[7,8]],[[7,8]],[[5,8],[6,8]],[[7,8]],[[7,8]],[[7]],[[8]]]

    #Test_Model.param_sampler = samplers.gauss(np.ones((2))*0.8, np.eye(2)*0.1)
    Test_Model.param_sampler = samplers.uniform(8, [0.2, 0.4, 0.3, 0.1, 0.7, 0.1, 0.2, 0.3]
                                                 , [0.3, 0.5, 0.5, 0.9, 0.9, 0.5, 1.0, 0.7])

    Test_Model.Labels = ["init", "reached"]
    Test_Model.Labelled_states = [[0,1,2,3,4,5,6,7,8],[7]]

    Test_Model.Name = "test"
        
    Test_Model.Formulae = ["Pmax=? [ F \"reached\"]"]

    Test_Model.Enabled_actions = [[0,1],[0,1],[0],[0],[0,1],[0],[0],[0],[0]]

    return Test_Model
