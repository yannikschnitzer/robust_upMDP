import Markov.models as Markov
import numpy as np
import Models.trans_funcs as t_f
import Models.samplers as samplers

def get_robot(n=3, jan_speed=2):
    Model = Markov.upMDP()

    Model.States = np.array(range(jan_speed*(n**4)))
    #States : [(Robot_location, janitor_location)] ignore transmission for now
    Model.Actions = np.array(range(5))
    Model.Init_state = 0+(n**2)//2
    
    Model.Labels = ["init", "crashed", "reached"]
    Model.Labelled_states = [[Model.Init_state],[],[]]

    Model.Enabled_actions = [[] for s in Model.States]
    Model.trans_ids = []
    Model.Transition_probs = []
    Model.paramed = []

    one = t_f.fixed(1)

    Model.max_supports = 0
    for s in Model.States:
        s_trans_ids = []
        s_trans_probs = []
        s_paramed = []
        if s < n**4:
            robot = s//n**2
            jan = s-robot*n**2
        else:
            robot = (s-n**4)//n**2
            jan = (s-n**4)-robot*n**2


        r_pos = (robot-n*(robot//n), robot//n)
        j_pos = (jan-n*(jan//n), jan//n)
        if r_pos != j_pos: # not crashed
            if r_pos != (n-1, n-1): # not reached goal
                Model.max_supports += 1
                for a in Model.Actions:
                    valid = False
                    if s < n**4: # if can_move
                        if a == 0:
                            # stay
                            s_base = s
                            valid = True
                        elif a == 1:
                            # move right
                            if r_pos[0] < n-1:
                                valid=True
                                s_base = s+n**2
                        elif a == 2:
                            # move left
                            if r_pos[0] > 0:
                                valid=True
                                s_base = s-n**2
                        elif a == 3:
                            # move up
                            if r_pos[1] < n-1:
                                valid=True
                                s_base = s+n**3
                        elif a == 4:
                            # move down
                            if r_pos[1] > 0:
                                valid=True
                                s_base = s-n**3
                        s_base += (jan_speed-1)*n**4
                    else:
                        if a == 0:
                            valid = True
                            s_base = s - n**4


                    if valid:
                        Model.Enabled_actions[s].append(a)
                        s_a_trans_ids = [s_base]
                        s_a_trans_probs = ["fill"]
                        s_a_paramed = [True]
                        filled = []
                        if j_pos[0] > 0:
                            s_a_paramed.append(True)
                            filled.append(1)
                            s_a_trans_probs.append(t_f.softmax_multi(1))
                            s_a_trans_ids.append(s_base-1)
                        if j_pos[0] < n-1:
                            s_a_paramed.append(True)
                            filled.append(2)
                            s_a_trans_probs.append(t_f.softmax_multi(2))
                            s_a_trans_ids.append(s_base+1)
                        if j_pos[1] > 0:
                            s_a_paramed.append(True)
                            filled.append(3)
                            s_a_trans_ids.append(s_base-n)
                            s_a_trans_probs.append(t_f.softmax_multi(3))
                        if j_pos[1] < n-1:
                            s_a_paramed.append(True)
                            filled.append(4)
                            s_a_trans_ids.append(s_base+n)
                            s_a_trans_probs.append(t_f.softmax_multi(4))
                        unfilled = [i for i in range(5) if i not in filled]
                        s_a_trans_probs[0] = t_f.softmax_filler(unfilled)
                        s_trans_ids.append(s_a_trans_ids)
                        s_trans_probs.append(s_a_trans_probs)
                        s_paramed.append(s_a_paramed)
            else:
                Model.Enabled_actions[s] = [0]
                Model.Labelled_states[2].append(s)
                s_trans_ids.append([s])
                s_trans_probs.append([one])
                s_paramed.append([False])
        else:
            s_paramed.append([False])
            Model.Enabled_actions[s] = [0]
            Model.Labelled_states[1].append(s)
            s_trans_probs.append([one])
            s_trans_ids.append([s])
        Model.trans_ids.append(s_trans_ids)
        Model.Transition_probs.append(s_trans_probs)
        Model.paramed.append(s_paramed)
    Model.Name = "Robot_model"
        
    Model.Formulae = ["Pmax=? [ F \"reached\"]"]
    
    #Hol_Model.param_sampler = samplers.gauss(np.array([0.5]), np.array([[0.2]]))
    Model.param_sampler = samplers.uniform(5, [1, 1, 1, -100, 1], 
                                              [10, 10, 10, 100, 10])
    return Model


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

