import Markov.models as Markov
import numpy as np
import Models.trans_funcs as t_f
import Models.samplers as samplers

def get_model(num_states, num_acts=2):
    Test_Model = Markov.upMDP()

    Test_Model.States = np.array(range(num_states+2))
    Test_Model.Actions = np.array(range(num_acts))
    Test_Model.Init_state = 0
    
    zero = t_f.fixed(0.0)
    one = t_f.fixed(1.0)
    
    Test_Model.Transition_probs = [[[] for a in range(num_acts)] for s in range(num_states)]
    Test_Model.trans_ids = [[[] for a in range(num_acts)] for s in range(num_states)]
    Test_Model.paramed = [[[] for a in range(num_acts)] for s in range(num_states)]
    Test_Model.Enabled_actions = [[a for a in range(num_acts)] for s in range(num_states)]

    funcs = [t_f.linear_multi(i) for i in range(num_states*num_acts)]
    
    for s in range(num_states):
        for a in range(num_acts):
            if s+a+1 < num_states+1:
                Test_Model.trans_ids[s][a] = [s+a+1, num_states+1]
            else:
                Test_Model.trans_ids[s][a] = [num_states, num_states+1]
            Test_Model.Transition_probs[s][a] = funcs[s*num_acts+a]
            Test_Model.paramed[s][a] = [True,True]

    Test_Model.trans_ids.append([[num_states]])
    Test_Model.trans_ids.append([[num_states+1]])
    Test_Model.Transition_probs.append([[one]])
    Test_Model.Transition_probs.append([[one]])
    Test_Model.paramed.append([[False]])
    Test_Model.paramed.append([[False]])
    Test_Model.Enabled_actions.append([0])
    Test_Model.Enabled_actions.append([0])

    Test_Model.max_supports = num_states*num_acts

    #Test_Model.param_sampler = samplers.gauss(np.array([0.5]), np.array([[0.2]]))
    Test_Model.param_sampler = samplers.gauss(np.ones((num_states*num_acts))*0.5, np.eye(num_states*num_acts)*0.3)
    
    Test_Model.Labels = ["init", "reached"]
    Test_Model.Labelled_states = [[0],[num_states]]

    Test_Model.Name = "expander"
        
    Test_Model.Formulae = ["Pmax=? [ F \"reached\"]"]


    return Test_Model
