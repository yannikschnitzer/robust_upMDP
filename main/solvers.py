import numpy as np
import cvxpy as cp
import Markov.writer as writer
import Markov.models
from PAC.funcs import *
import itertools
from tqdm import tqdm
import time
import copy
import pickle
import logging
import os

def load_samples(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data[0], data[1]

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def save_data(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def get_samples(model, N):
    samples = [model.param_sampler() for j in range(N)]
    return samples

def test_pol(model, samples, pol=None, paramed_models = None):
    num_states = len(model.States)
    num_acts = len(model.Actions)
     
    true_probs = []
    pols = []
    if model.opt == "max":
        wc = 1
    else:
        wc = 0
    for ind, sample in enumerate(samples):
        time_start = time.perf_counter()
        if paramed_models is None:
            test_MDP = model.fix_params(sample)
        else:
            test_MDP = paramed_models[ind]
        time_fix_params = time.perf_counter()
        if pol is not None:
            test_model = test_MDP.fix_pol(pol)
        else:
            test_model = test_MDP
        time_fix_pol = time.perf_counter()
        IO = writer.stormpy_io(test_model)
        IO.write()
        res, all_res, sol_pol = IO.solve()
        pols.append(sol_pol)
        if model.opt == "max":
            if wc>res[0]:
                wc = res[0]
        else:
            if wc<res[0]:
                wc = res[0]
        true_probs.append(all_res[0])

        time_end = time.perf_counter()
        time_for_fixing_param = time_fix_params -time_start
        time_for_fixing_pol = time_fix_pol -time_fix_params
        time_for_solving = ((time_end-time_fix_pol))
        logging.debug("solving {:.3f}s".format(time_for_solving))
        logging.debug("fixing params {:.3f}s".format(time_for_fixing_param))
        logging.debug("fixing pol {:.3f}s".format(time_for_fixing_pol))
    true_probs = np.array(true_probs)
    return wc, true_probs, pols

def solve_subgrad(samples, model, max_iters=500):
    print("--------------------\nStarting subgradient descent")
    fixed_MDPs = [model.fix_params(sample) for sample in samples]

    num_states = len(model.States)
    num_acts = len(model.Actions)
    pol = np.zeros((num_states, num_acts))
    for s in model.States:
        for a in model.Enabled_actions[s]:
            pol[s,a] = 1/len(model.Enabled_actions[s])

    projected = cp.Variable(num_acts)
    point = cp.Parameter(num_acts)

    obj = cp.Minimize(cp.norm(projected-point))
    wc_hist = []
    wc, true_probs, _ = test_pol(model, samples, pol, paramed_models = fixed_MDPs)
    worst = np.argwhere(true_probs[:,model.Init_state]==wc)
    worst = np.random.choice(worst.flatten())
    for i in tqdm(range(max_iters)):
        time_start = time.perf_counter()
        
        step = 1/(i+1)
        step = 0.1
        grad = np.zeros_like(pol)
        for s in model.States:
            if len(model.Enabled_actions[s]) > 1:
                for a in model.Enabled_actions[s]:
                    grad_finder = np.copy(pol)
                    grad_finder[s] = 0
                    grad_finder[s,a] = 1
                    tic = time.perf_counter()
                    grad[s,a] = test_pol(model, 
                                         [samples[worst]], 
                                         grad_finder, 
                                         paramed_models = [fixed_MDPs[worst]])[0] 
                    toc = time.perf_counter()
                    #print(toc-tic)
        time_grads = time.perf_counter()-time_start
        logging.debug("Total time for finding gradients: {:.3f}".format(time_grads))
        pol += step*grad
        
        for s in model.States:
            if len(model.Enabled_actions[s]) <= 1:
                pass
            elif len(model.Enabled_actions[s]) == 2:
                act_0 = model.Enabled_actions[s][0]
                act_1 = model.Enabled_actions[s][1]
                diff = pol[s,act_0]-pol[s,act_1]
                if diff > 1:
                    pol[s,act_0] = 1
                    pol[s,act_1] = 0
                elif diff < -1:
                    pol[s, act_0] = 0
                    pol[s, act_1] = 1
                else:
                    pol[s,act_0] = (1+diff)/2
                    pol[s,act_1] = (1-diff)/2
            else:
                cons = [cp.norm(projected,1) <= 1, projected >= 0]
                for a in range(num_acts):
                    if a not in model.Enabled_actions[s]:
                        cons += [projected[a] == 0]
                point.value = pol[s]
                prob = cp.Problem(obj, cons)
                res = prob.solve()
                pol[s] = projected.value
        time_proj = time.perf_counter()-time_start-time_grads
        logging.debug("Time for projection step: {:.3f}".format(time_proj))
        wc, true_probs, _ = test_pol(model, samples, pol)
        worst = np.argwhere(true_probs[:,model.Init_state]==wc)
        worst = np.random.choice(worst.flatten())
        wc_hist.append(wc)
    return wc, pol

def find_all_pols(model):
    base_vec = [False for i in model.Actions]
    act_vecs = []
    
    print("--------------------\nFinding policies")
    for a in model.Actions:
        vec = copy.copy(base_vec)
        vec[a] = True
        act_vecs.append(vec)
    act_poss = [[copy.copy(act_vecs[a]) for a in model.Enabled_actions[s]] for s in model.States]
    print("Built list, now building full policies")
    pols = [np.array(pol) for pol in itertools.product(*act_poss)]
    return pols

def build_init_payoff(samples, model):
    pols = find_all_pols(model)
    all_probs = []
    non_dommed_pols = []
    print("--------------------\nBuilding initial payoff matrix")
    best_wc = 0
    for j, pol in enumerate(tqdm(pols)):
        # could also find best deterministic policy here
        probs = test_pol(model, samples, pol)[1][:,model.Init_state]
        if model.opt == "max":
            wc = min(probs)
            if wc > best_wc:
                best_wc = wc
                det_pol = pol
        else:
            wc = max(probs)
            if wc < best_wc:
                best_wc = wc
                det_pol = pol
                
        dommed = False
        for i, elem in enumerate(all_probs):
            if model.opt == "max":
                if np.all(probs <= elem):
                    dommed = True
                elif np.all(probs >= elem):
                    all_probs.pop(i)
                    non_dommed_pols.pop(i)
            else:
                if np.all(probs >= elem):
                    dommed = True
                elif np.all(probs <= elem):
                    all_probs.pop(i)
                    non_dommed_pols.pop(i)
        if not dommed:
            all_probs.append(probs)
            non_dommed_pols.append(j)
    pols = [pols[i] for i in non_dommed_pols]
    payoffs = np.vstack(all_probs)
    return payoffs, pols

def calc_payoff_mat(samples, model):
    payoffs, pols = build_init_payoff(samples, model)

    print("--------------------\nRemoving dominated samples")
    
    non_domed_samples = []
    for i in tqdm(range(len(samples))):
        if model.opt == "max":
            test_arr = np.all(payoffs[:,i][:,np.newaxis] > payoffs, axis=0)
        else:
            test_arr = np.all(payoffs[:,i][:,np.newaxis] < payoffs, axis=0)
        if not np.any(test_arr):
            non_domed_samples.append(i)
    non_domed_payoffs = payoffs[:, non_domed_samples]
    return non_domed_payoffs, pols, non_domed_samples

def MNE_solver(samples, model):
    payoffs, pols, rel_samples = calc_payoff_mat(samples, model)
    best = 0
    print("---------------------\nIterating through MNE combinations")
    combs = list(itertools.combinations(range(len(rel_samples)), len(pols)))
    for elem in tqdm(combs):
        test_non = elem
        test_non_domed_payoffs = payoffs[:, test_non]
        br_mixer = np.ones((1, len(pols)))@np.linalg.inv(test_non_domed_payoffs)
        br_mixer = np.maximum(0, br_mixer)
        br_mixer /= np.sum(br_mixer)
        res = (br_mixer@payoffs).flatten()
        if min(res) > best:
            best = min(res)
            best_samples = elem
            best_pol = np.copy(br_mixer)
    return best, best_pol, pols 

def FSP_solver(samples, model, max_iters = 100000):
    update_parallel = True
    payoffs, pols, rel_samples = calc_payoff_mat(samples, model)
    pol_dist = np.ones((1,len(pols)))/len(pols)
    sample_dist = np.ones((len(rel_samples),1))/len(rel_samples)

    sum_step = 0
    print("---------------------\nStarting FSP")
    for i in tqdm(range(max_iters)):
        #step = 1/(i+1)
        step = 1
        sum_step += step
        new = step/sum_step
        old = 1-new
        if update_parallel:
            sample_vec = pol_dist@payoffs
            pol_vec = payoffs@sample_dist
            
            pol_dist *= old
            pol_dist[0, np.argmax(pol_vec)] += new
            
            sample_dist *= old
            sample_dist[np.argmin(sample_vec), 0] += new
            
    res = min((pol_dist@payoffs).flatten())
    return res, pol_dist, pols, np.sum(sample_dist >= 1e-4)

def run_all(args):
    print("Running code for robust optimal policy \n --------------------")
    model = args["model"]
    #test_support_num(args)
    #a_priori_max_supports = sum([len(acts) for acts in model.Enabled_actions])
    a_priori_max_supports = model.max_supports
    #calc_max_path(model)
    a_priori_eps = calc_eps(args["beta"], args["num_samples"], a_priori_max_supports)
    
    print("A priori upper bound on number of support constraints is " + str(a_priori_max_supports))

    print("A priori bound on violation probability is {:.3f} with confidence {:.3f}".format(a_priori_eps, args["beta"]))

    if args["sample_load_file"] is not None:
        base, samples = load_samples(args["sample_load_file"])
    else:
        samples = get_samples(model, args["num_samples"])
    if args["sample_save_file"] is not None:
        save_data(args["sample_save_file"], (base, samples))
    
    if args["prob_load_file"] is not None:
        warm_probs = load_data(args["prob_load_file"])
    else:
        warm_probs = None
    num_states = len(model.States)
  
    res_sg, pol_sg = solve_subgrad(samples, model)
    import pdb; pdb.set_trace()
    res_MNE, pol_MNE, pols = MNE_solver(samples, model)
    res_FSP, pol_FSP, pols, a_post_support_num = FSP_solver(samples, model)

    wc, probs, _ = test_pol(model, samples, pol_sg)
    supports_2 = np.sum(probs[:, model.Init_state] <= wc+1e-3)
    assert supports_2 == a_post_support_num

    print("----------------\nResults:\nmatrix solver: {:.3f}\nFSP: {:.3f}\nSubgradient: {:.3f}".format(res_MNE, res_FSP, res_sg))

    if args["result_save_file"] is not None:
        save_data(args["result_save_file"], {"worst": wc, "probs":all_p, "pol":pol, "supports":supports})

    N = args["num_samples"]

    [a_post_eps_L, a_post_eps_U] = \
        calc_eps_risk_complexity(args["beta"], N, a_post_support_num)
    
    print("A posteriori, found " + str(a_post_support_num) + " support constraints")

    print("A posteriori, violation probability is in the range [{:.3f}, {:.3f}], with confidence {:.3f}"
          .format(a_post_eps_L, a_post_eps_U, args["beta"]))

    print("Optimal satisfaction probability is found to be {:.3f}".format(res_sg))

    #if pol.size < 50:
    #    print("Calculated robust policy is:")
    #    print(pol)

    #thresh = a_priori_eps

    if args["MC"]:
        emp_violation = MC_sampler(model, args["MC_samples"], res_sg, pol_sg) 
        print("Empirical violation rate is found to be {:.3f}".format(emp_violation))
    print("\n\n")

def test_support_num(args):
    model = args["model"]
    num_states = len(model.States)
    num_acts = len(model.Actions)
    for i in range(10):
        base, samples = gen_samples(model, args["num_samples"], args["batch_size"])

        probs, pol, supports, a_post_support_num, all_p = calc_probs_policy_iteration(model, base, samples)
         
        print("Calculated supports: " + str(supports))
        actual_supports = []
        for j in range(args["num_samples"]):
            samples_new = remove_sample(j, samples, num_states, num_acts)
            test_probs, test_pol, _, _, all_p_test = calc_probs_policy_iteration(model, base, samples_new)
            if model.opt == "max":
                gap = abs(min(probs) - min(test_probs))
            else:
                gap = abs(max(probs) - max(test_probs))
            if gap >= 1e-4:
                actual_supports.append(j)
        print("Emprical supports: " + str(actual_supports))
        if len(actual_supports) > model.max_supports:
            print("ERROR, a priori max sc was wrong!!!")
            import pdb; pdb.set_trace()
        for sc in actual_supports:
            if sc not in supports:
                print("ERROR, found an empirical SC not in a posteriori support set")
                import pdb; pdb.set_trace()

def remove_sample(pos, samples):
    new_samples = copy.copy(samples)
    new_samples.pop(pos)
    return new_samples
