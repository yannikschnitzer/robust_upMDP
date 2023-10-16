import numpy as np
import cvxpy as cp
import Markov.writer as writer
import Markov.models
from PAC.funcs import *
import itertools
from tqdm import tqdm
import time
import copy
import logging
import os
import matplotlib.pyplot as plt
import pycarl
from main.sampler import *
import datetime

def check_timeout(start, max_time=3600):
    if time.perf_counter() - start > max_time:
        print("Timed out!")
        return True
    else:
        return False

def test_pol(model, samples, pol=None, paramed_models = None):
    num_states = len(model.States)
    num_acts = len(model.Actions)
    
    if paramed_models is not None:
        test_MDP = model.fix_params(samples[0])

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
            test_MDP.Transition_probs = paramed_models[ind]
        time_fix_params = time.perf_counter()
        if pol is not None:
            test_model = test_MDP.fix_pol(pol)
        else:
            test_model = test_MDP
        time_fix_pol = time.perf_counter()
        IO = writer.stormpy_io(test_model)
        #IO = writer.PRISM_io(test_model)
        IO.write()
        time_write = time.perf_counter()

        res, all_res, sol_pol = IO.solve()
        pols.append(sol_pol)
        if model.opt == "max":
            if wc>res[0]:
                wc = res[0]
        else:
            if wc<res[0]:
                wc = res[0]
        true_probs.append(all_res) # all_res[0] if using stormpy?

        time_end = time.perf_counter()
        time_for_fixing_param = time_fix_params -time_start
        time_for_fixing_pol = time_fix_pol -time_fix_params
        time_for_writing = time_write - time_fix_pol
        time_for_solving = ((time_end-time_write))
        logging.debug("solving {:.3f}s".format(time_for_solving))
        logging.debug("fixing params {:.3f}s".format(time_for_fixing_param))
        logging.debug("writing {:.3f}s".format(time_for_writing))
        logging.debug("fixing pol {:.3f}s".format(time_for_fixing_pol))
    true_probs = np.array(true_probs)
    return wc, true_probs, pols

def find_grad(model, pol, worst_sample):
    grad = np.zeros_like(pol)
    norm = 0
    time_sum = 0
    #min_elem = 1 
    #max_elem = 0
    test_MDP = model.fix_params(worst_sample)
    nom_MC = test_MDP.fix_pol(pol)
    nom_ids = copy.copy(nom_MC.trans_ids)
    nom_probs = copy.copy(nom_MC.Transition_probs)
    for s in model.States:
        if len(model.Enabled_actions[s]) > 1:
            nom_MC.trans_ids = copy.copy(nom_ids)
            nom_MC.Transition_probs = copy.copy(nom_probs)
            for a in model.Enabled_actions[s]:
                tic = time.perf_counter()
                grad_finder = np.zeros(len(model.Actions))
                grad_finder[a] = 1
                s_primes, s_probs = test_MDP.fix_state_pol(grad_finder, s)
                nom_MC.trans_ids[s] = s_primes
                nom_MC.Transition_probs[s] = s_probs
                IO = writer.stormpy_io(nom_MC)
                #IO = writer.PRISM_io(test_model)
                IO.write()
                time_write = time.perf_counter()

                res, _, _ = IO.solve()
                grad[s,a] = res[0]
                toc = time.perf_counter()
                #logging.debug("Time to find gradient: " + str(toc-tic))
    pre_norming = time.perf_counter()
    grad /= np.linalg.norm(grad, ord="fro")
    time_norming = time.perf_counter()-pre_norming
    logging.debug("time for fro: "+str(time_norming))
    
    return grad

def solve_subgrad(samples, model, max_iters=500, quiet=False, tol=1e-3, init_step=1, step_exp=1/2):
    start = time.perf_counter()
    if not quiet:
        print("--------------------\nStarting subgradient descent")
   
    if quiet:
        def tqdm(item):
            return item
    else:
        from tqdm import tqdm
    sample_trans_probs = []
    for sample in samples:
        new_MDP = model.fix_params(sample)
        sample_trans_probs.append(copy.copy(new_MDP.Transition_probs))

    arr = model.get_trans_arr()

    num_states = len(model.States)
    num_acts = len(model.Actions)
    pol = np.zeros((num_states, num_acts))
    for s in model.States:
        for a in model.Enabled_actions[s]:
            pol[s,a] = 1/len(model.Enabled_actions[s])
    
    wc_hist = []
    best_hist = []
    tic = time.perf_counter()
    wc, true_probs, _ = test_pol(model, samples, pol, paramed_models = sample_trans_probs)
    worst = np.argwhere(true_probs[:,model.Init_state]==wc)
    worst = np.random.choice(worst.flatten())
    toc = time.perf_counter()
    logging.debug("Time for finding worst case: {:.3f}s".format(toc-tic)) # This is also done every iteration, could be sped up but takes ~6/1500 the time
    best_worst_pol = test_pol(model, [samples[worst]])[2][0]
    test_wc, test_probs, _ = test_pol(model, samples, best_worst_pol, paramed_models = sample_trans_probs)
    test_worst = np.argwhere(test_probs[:,model.Init_state]==test_wc).flatten()
    if worst in test_worst:
        info = {"hist":[test_wc], "all":test_probs[:, model.Init_state]}
        if not quiet:
            print("Worst case holds with deterministic policy, deterministic is optimal")
        return test_wc, best_worst_pol, test_worst, info
    pol = 0.1*pol + 0.9*best_worst_pol # a nicer start point
    wc, true_probs, _ = test_pol(model, samples, pol, paramed_models = sample_trans_probs)
    worst = np.argwhere(true_probs[:,model.Init_state]==wc)
    worst = np.random.choice(worst.flatten())

    if model.opt == "max":
        best = 0
    else:
        best = 1

    for i in tqdm(range(max_iters)):
        if check_timeout(start):
            return -1, None, None, None
        time_start = time.perf_counter()
        
        old_pol = np.copy(pol)
        step = init_step/((i+1)**step_exp)
        #step = 1/((i+1)**(1/2))
        #step = 0.1
        grad = find_grad(model, pol, samples[worst])
        #grad_norm = np.linalg.norm(grad, ord=np.inf)
         
        time_grads = time.perf_counter()-time_start
        #if time_grads > 2:
        #    import pdb; pdb.set_trace()
        logging.debug("Total time for finding gradients: {:.3f}".format(time_grads))
        if model.opt == "max":
            pol += step*grad
        else:
            pol -= step*grad 

        for s in model.States:
            fin = False
            if len(model.Enabled_actions[s]) <= 1:
                pass
            else:
                acts = model.Enabled_actions[s]
                normal = np.ones(len(acts))/len(acts)**(1/2)
                orig = np.ones(len(acts))/len(acts)
                pol_s_a = pol[s][acts]
                v = pol_s_a - orig
                dist = np.dot(v,normal)
                proj = pol_s_a - dist*normal
                while True:
                    normal = np.ones(len(acts))/len(acts)**(1/2)
                    orig = np.ones(len(acts))/len(acts)
                    pol_s_a = pol[s][acts]
                    v = pol_s_a - orig
                    dist = np.dot(v,normal)
                    proj = pol_s_a - dist*normal
                    if np.all(proj > 0):
                        break
                    else:
                        pos = np.where(proj>0)[0]
                        acts = [acts[p] for p in pos]
                pol[s] = 0
                pol[s][acts] = proj
        time_proj = time.perf_counter()-time_start-time_grads
        logging.debug("Time for projection step: {:.3f}".format(time_proj))
        wc, true_probs, _ = test_pol(model, samples, pol, paramed_models = sample_trans_probs)
        worst = np.argwhere(true_probs[:,model.Init_state]==wc)
        worst = np.random.choice(worst.flatten())
        wc_hist.append(wc)
        if model.opt == "max":
            if wc > best:
                best = wc
                best_pol = pol
        else:
            if wc < best:
                best = wc
                best_pol = pol
        best_hist.append(best)
        logging.info("Iteration: {}".format(i+1))
        logging.info("Current value: {:.6f}, with sample {}".format(wc, worst))
        #logging.info("Policy inf norm change: {:.3f}".format(np.linalg.norm(pol-old_pol, ord=np.inf)))
        if len(wc_hist) >= 2:
            change=abs(wc_hist[-2]-wc_hist[-1])
            logging.info("Value change: {:.6f}".format(change))
            if change < tol:
                break
    final_time = time.perf_counter() - start
    print("Subgradient Descent took {:.3f}s".format(final_time))
    wc, true_probs, _ = test_pol(model, samples, best_pol, paramed_models = sample_trans_probs)
    #supp_tol=0.05
    #supp_tol =55 0*tol
    supp_tol=0.05 #conservative but works
    if model.opt == "max":
        active_sg = np.argwhere(true_probs[:, model.Init_state] <= wc+supp_tol) # this tol is hard to tune...
    else:
        active_sg = np.argwhere(true_probs[:, model.Init_state] >= wc-supp_tol)
    info = {"hist":best_hist, "all":true_probs[:, model.Init_state]}

    return best, best_pol, active_sg, info

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
                if np.all(probs < elem):
                    dommed = True
                elif np.all(probs > elem):
                    all_probs.pop(i)
                    non_dommed_pols.pop(i)
            else:
                if np.all(probs > elem):
                    dommed = True
                elif np.all(probs < elem):
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

def Feas_prog(payoffs, S_pol, S_adv):
    all_pol = list(range(payoffs.shape[0]))
    all_adv = list(range(payoffs.shape[1]))
    
    neg_S_pol = [elem for elem in all_pol if elem not in S_pol]
    neg_S_adv = [elem for elem in all_adv if elem not in S_adv]

    pol_strat = cp.Variable(payoffs.shape[0])
    adv_strat = cp.Variable(payoffs.shape[1])
    val = cp.Variable(2)
    objective = cp.Minimize(np.ones((1,2))@val)
    
    S_pol_vec = np.zeros(payoffs.shape[0])
    for i in S_pol:
        S_pol_vec[i] = 1
    S_adv_vec = np.zeros(payoffs.shape[1])
    for i in S_adv:
        S_adv_vec[i] = 1
    neg_S_pol_vec = np.zeros(payoffs.shape[0])
    for i in neg_S_pol:
        neg_S_pol_vec[i] = 1
    neg_S_adv_vec = np.zeros(payoffs.shape[1])
    for i in neg_S_adv:
        neg_S_adv_vec[i] = 1
    constraints = [
                    pol_strat >= 0,
                    adv_strat >= 0,
                    sum(pol_strat) == 1,
                    sum(adv_strat) == 1,
                    pol_strat@S_pol_vec == 1,
                    adv_strat@S_adv_vec == 1,
                    ]
    for i in S_adv:
        constraints.append((pol_strat@payoffs)[i] == val[1])
    for i in S_pol:
        constraints.append((payoffs@adv_strat)[i] == val[0])
    for i in neg_S_adv:
        constraints.append((pol_strat@payoffs)[i] >= val[1])
    for i in neg_S_pol:
        constraints.append((payoffs@adv_strat)[i] <= val[0])
    prob = cp.Problem(objective, constraints)
    try:
        result = prob.solve()
    except cp.error.SolverError:
        return None, None
    except:
        print("UNEXPECTED ERROR IN CVXPY")
        return None, None
    if result is not np.inf:
        return (pol_strat.value, adv_strat.value), val.value[0]
    else:
        return None, None

def PNS_algo(payoffs):
    start = time.perf_counter()
    strat_lengths = payoffs.shape
    x_1_poss = list(range(1, strat_lengths[0]+1))
    x_2_poss = list(range(1, strat_lengths[1]+1))
    comb_poss = []
    max_sum = 0
    for x_1 in x_1_poss:
        for x_2 in x_2_poss:
            comb_poss.append((x_1, x_2))
            max_sum = max(max_sum, x_1+x_2)
    sorter = lambda e: np.abs(e[0]-e[1])*(max_sum+1)+e[0]+e[1]
    comb_poss.sort(key=sorter)
    for x_1, x_2 in tqdm(comb_poss):
        if check_timeout(start):
            return None, None
        supps_pols = list(itertools.combinations(range(strat_lengths[0]), x_1))
        for S_pol in supps_pols:
            if check_timeout(start):
                return None, None
            redux_payoffs = payoffs[S_pol, :]
            Adv_act_prime = []
            for i, elem in enumerate(redux_payoffs.T):
                if check_timeout(start):
                    return None, None
                if not np.any(np.all(elem.T[:,np.newaxis] > redux_payoffs, axis=0)):
                    # elem not dominated
                    Adv_act_prime.append(i)
            redux_redux_payoffs = redux_payoffs[:, Adv_act_prime]
            conditional_dommed_exists = False
            for i, elem in enumerate(redux_redux_payoffs):
                if check_timeout(start):
                    return None, None
                if np.any(np.all(elem[np.newaxis, :] < redux_redux_payoffs, axis=1)):
                    conditional_dommed_exists = True
            if not conditional_dommed_exists:
                supps_adv = list(itertools.combinations(Adv_act_prime, x_2))
                for S_adv in supps_adv:
                    if check_timeout(start):
                        return None, None
                    r_r_r_payoffs = redux_payoffs[:, S_adv]
                    conditional_dommed_exists_2 = False
                    for i, elem in enumerate(r_r_r_payoffs):
                        if check_timeout(start):
                            return None, None
                        if np.any(np.all(elem[np.newaxis, :] < r_r_r_payoffs, axis=1)):
                            conditional_dommed_exists_2 = True
                    if not conditional_dommed_exists_2:
                        pol, val = Feas_prog(payoffs, S_pol, S_adv)
                        if pol is not None:
                            return pol, val

def MNE_solver(samples, model):
    payoffs, pols, rel_samples = calc_payoff_mat(samples, model)
    if model.opt != "max":
        payoffs = -payoffs
    pol, val = PNS_algo(payoffs)
    if pol is not None:
        info = {"pols": pol, "all":(pol[0]@payoffs).flatten(), "ids":rel_samples}
        pol = pol[0]
    else:
        info = None
    if val is None:
        val = -1
    supps = np.sum(pol >= 1e-5)
    return val, pol, supps, info

def det_maxmin(samples, model):
    start = time.perf_counter()
    payoffs, pols, rel_samples = calc_payoff_mat(samples, model)
    if model.opt != "max":
        payoffs = -payoffs
    min_vals = np.min(payoffs, axis=1)
    best = np.argmax(min_vals)
    pol = pols[best]
    val = min_vals[best]
    val_loc = np.argwhere(payoffs[best,:] == val)
    supps = np.argwhere(payoffs[:,val_loc] >= val) #(-1 to remove val==val)
    supps = supps[:,0].tolist()
    supps.pop(supps.index(best))

    supps_ub = len(supps)
    
    block_samples = []
    block_set = set()
    for supp in supps:
        poss_samples = np.argwhere(payoffs[supp,:] <= val)
        curr_samples = poss_samples.flatten().tolist()
        block_samples.append(curr_samples)
        block_set  = block_set.union(curr_samples)
    for i in range(len(block_set)):
        if check_timeout(start):
            return -1, None, None, None
        test_sets = list(itertools.combinations(block_set, i+1))
        for test in test_sets:
            hitter = True
            for elem in block_samples:
                intersect = set(test).intersection(elem)
                if len(intersect) == 0:
                    hitter = False
                    break
            if hitter:
                hit_set = test
                break
        if hitter:
            break
    supps = len(hit_set)

    info = {"pols":pol, "all":(payoffs[best,:]).flatten(), "ids":rel_samples}

    return val, pol, supps, info


def FSP_solver(samples, model, max_iters = 100000):
    start = time.perf_counter()
    update_parallel = True
    payoffs, pols, rel_samples = calc_payoff_mat(samples, model)
    pol_dist = np.ones((1,len(pols)))/len(pols)
    sample_dist = np.ones((len(rel_samples),1))/len(rel_samples)

    sum_step = 0
    print("---------------------\nStarting FSP")
    for i in tqdm(range(max_iters)):
        if check_timeout(start):
            return -1, None, None, None
        #step = 1/(i+1)
        step = 1
        sum_step += step
        new = step/sum_step
        old = 1-new
        if update_parallel:
            sample_vec = pol_dist@payoffs
            pol_vec = payoffs@sample_dist
            
            pol_dist *= old
            sample_dist *= old
            if model.opt == "max":
                pol_dist[0, np.argmax(pol_vec)] += new
                sample_dist[np.argmin(sample_vec), 0] += new
            else:
                pol_dist[0, np.argmin(pol_vec)] += new
                sample_dist[np.argmax(sample_vec), 0] += new


    if model.opt == "max":
        res = min((pol_dist@payoffs).flatten())
    else:
        res = max((pol_dist@payoffs).flatten())

    info = {"pols": pols, "all":(pol_dist@payoffs).flatten(), "ids":rel_samples}
    return res, pol_dist, np.sum(sample_dist >= 1e-4), info

def run_all(args, samples):
    start = datetime.datetime.now().isoformat().split('.')[0]
    print("Running code for robust optimal policy \n --------------------")
    model = args["model"]
    if args["test_supps"]:
        test_support_num(args)
    else:
        a_priori_max_supports = model.max_supports
        a_priori_eps = calc_eps(args["beta"], args["num_samples"], a_priori_max_supports)
        
        print("A priori upper bound on number of support constraints is " + str(a_priori_max_supports))

        print("A priori bound on violation probability is {:.3f} with confidence {:.3f}".format(a_priori_eps, args["beta"]))
 
        if args["prob_load_file"] is not None:
            warm_probs = load_data(args["prob_load_file"])
        else:
            warm_probs = None
        num_states = len(model.States)

        ## subgradient

        N = args["num_samples"]
        start_time = time.perf_counter()
        res_sg, pol_sg, active_sg, info_sg = solve_subgrad(samples, model, max_iters=args["sg_itts"], tol=args["tol"], init_step=args["init_step"], step_exp=args["step_exp"])

        if pol_sg is not None: 
            sg_time = time.perf_counter()-start_time
            sg_active_num = active_sg.size 
            if args["save_figs"] or args["output_figs"]:
                res_plot = [res_sg - i for i in info_sg["hist"]]
                res_plot.pop(-1)
                fig, ax = plt.subplots()
                ax.loglog(res_plot)
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Distance from final satisfaction probability")
       

            if args["save_figs"]:
                fname = "plots/" + start + 'dist_fig'
                plt.savefig(fname + ".png", bbox_inches="tight")
                plt.savefig(fname + ".pdf", bbox_inches="tight")
            elif args["output_figs"]:
                plt.show()
            
            if args["save_figs"] or args["output_figs"]:
                fig2, ax2 = plt.subplots()
                ax2.loglog(info_sg["hist"])
                ax2.set_xlabel("Iteration")
                ax2.set_ylabel("Satisfaction probability")
            
            if args["save_figs"]:
                fname = "plots/" + start + 'prob_fig'
                plt.savefig(fname + ".png", bbox_inches="tight")
                plt.savefig(fname + ".pdf", bbox_inches="tight")
            elif args["output_figs"]:
                plt.show()
            
            if args["MC"]:
                emp_violation = MC_sampler(model, args["MC_samples"], res_sg, pol_sg) 
                print("Empirical violation rate is found to be {:.3f}".format(emp_violation))
            if args["MC_pert"]:
                pert_violation = MC_perturbed(model, args["MC_samples"], res_sg, pol_sg) 
                print("Noisy violation rate is found to be {:.3f}".format(pert_violation))

            print("Using subgradient methods found " + str(active_sg.size) + " active constraints a posteriori")
            [a_post_eps_L, a_post_eps_U] = \
                calc_eps_risk_complexity(args["beta"], N, active_sg.size)

            print("Hence, a posteriori, violation probability is in the range [{:.3f}, {:.3f}], with confidence {:.3f}"
                    .format(a_post_eps_L, a_post_eps_U, args["beta"]))

            print("Optimal satisfaction probability is found to be {:.3f}".format(res_sg))
        else:
            sg_time = -1
        pols = {"subgradient":pol_sg}
        res = {"subgradient":res_sg}


        if len(model.States)**len(model.Actions) < 200:
            start_time = time.perf_counter()
            res_MNE, pol_MNE, MNE_support_num, info_MNE = MNE_solver(samples, model)
            if pol_MNE is not None:
                MNE_time = time.perf_counter()-start_time
                if args["MC"]:
                    emp_MNE = MC_sampler(model, args["MC_samples"], res_MNE, pol_MNE) 
                    print("Empirical violation rate is found to be {:.3f}".format(emp_MNE))
                if args["MC_pert"]:
                    pert_MNE = MC_perturbed(model, args["MC_samples"], res_MNE, pol_MNE)
                    print("Noisy violation rate is found to be {:.3f}".format(pert_MNE))
                print("Using PNS algo found " + str(MNE_support_num) + " support constraints a posteriori")
                [MNE_a_post_eps_L, MNE_a_post_eps_U] = \
                    calc_eps_risk_complexity(args["beta"], N, MNE_support_num)

                print("Hence, a posteriori, violation probability is in the range [{:.3f}, {:.3f}], with confidence {:.3f}"
                    .format(MNE_a_post_eps_L, MNE_a_post_eps_U, args["beta"]))
                print("\n------------------\n")

            else:
                MNE_time = -1

            start_time = time.perf_counter()
            res_FSP, pol_FSP, FSP_a_post_support_num, info_FSP = FSP_solver(samples, model, max_iters=args["FSP_itts"])
            if pol_FSP is not None:
                FSP_time = time.perf_counter()-start_time
                if args["MC"]:
                    emp_FSP = MC_sampler(model, args["MC_samples"], res_FSP, pol_FSP) 
                    print("Empirical violation rate is found to be {:.3f}".format(emp_violation))
                if args["MC_pert"]:
                    pert_FSP = MC_perturbed(model, args["MC_samples"], res_FSP, pol_FSP)
                [FSP_a_post_eps_L, FSP_a_post_eps_U] = \
                    calc_eps_risk_complexity(args["beta"], N, FSP_a_post_support_num)
                print("Using FSP algo found " + str(FSP_a_post_support_num) + " support constraints a posteriori")

                print("Hence, a posteriori, violation probability is in the range [{:.3f}, {:.3f}], with confidence {:.3f}"
                    .format(FSP_a_post_eps_L, FSP_a_post_eps_U, args["beta"]))
                print("\n------------------\n")
            else:
                FSP_time = -1

            start_time = time.perf_counter()
            res_det, pol_det, supps_det, info_det = det_maxmin(samples, model)
            if pol_det is not None:
                det_time = time.perf_counter()-start_time
                eps_det = calc_eps_nonconvex(args["beta"], N, supps_det)
                if args["MC"]:
                    emp_det = MC_sampler(model, args["MC_samples"], res_det, pol_det) 
                    print("Empirical violation rate is found to be {:.3f}".format(emp_violation))
                if args["MC_pert"]:
                    pert_det = MC_perturbed(model, args["MC_samples"], res_det, pol_det)
                print("Using deterministic policy found " + str(supps_det) + " support constraints a posteriori")

                print("Hence, a posteriori, violation probability is bounded by {:.3f}, with confidence {:.3f}"
                    .format(eps_det, args["beta"]))
                print("\n------------------\n")
            else:
                det_time = -1
            #if active_sg.size != a_post_support_num:
            #    print("Found {} supports using subgradient method, but {} using fictitious self play".format(active_sg.size, a_post_support_num))
            

            print("----------------\nResult comparison:\nPNS algo: {:.13f}\nDet.: {:.13f}\nFSP: {:.13f}\nSubgradient: {:.13f}".format(res_MNE, res_det, res_FSP, res_sg))
            res["MNE"] = res_MNE
            res["FSP"] = res_FSP
            res["det"] = res_det
            pols["MNE"] = pol_MNE
            pols["FSP"] = pol_FSP
            pols["det"] = pol_det
        else:
            MNE_time = -1
            FSP_time = -1
            det_time = -1

        if args["result_save_file"] is not None:
            save_data(args["result_save_file"], {"res": res, "pols":pols})

        if pol_sg.size < 50:
            print("Calculated robust policy using subgradient methods is:")
            print(pol_sg)

        thresh = a_priori_eps

        print("\n\n")
        return MNE_time, FSP_time, sg_time, det_time

def test_support_num(args):
    print("Running code to test number support set calculation\n--------------")
    model = args["model"]
    num_states = len(model.States)
    num_acts = len(model.Actions)
    num_over = 0
    num_errors = 0
    num_underestimates = 0
    for i in range(10):
        print("-----------------\nGenerating new sample batch")
        samples = get_samples(args)

        #probs, pol, supports, a_post_support_num, all_p = calc_probs_policy_iteration(model, base, samples)
        print("Solving for sample batch")
        #res_sg, pol_sg, active_sg, _ = solve_subgrad(samples, model, max_iters=100, quiet=True)
        res_sg, pol_sg, active_sg, _ = solve_subgrad(samples, model, max_iters=args["sg_itts"], tol=args["tol"], init_step=args["init_step"], step_exp=args["step_exp"])
        print("Calculated supports: " + str(active_sg+1)) # Add 1 for 1 indexign
        samples_new = [samples[int(j)] for j in active_sg]
        print("Solving for calculated supports only")
        test_res, test_pol, _, _ = solve_subgrad(samples_new, model, max_iters=args["sg_itts"], tol=args["tol"], init_step=args["init_step"], step_exp=args["step_exp"])
        #test_res, test_pol, _, _ = solve_subgrad(samples_new, models, max_it)

        gap = test_res-res_sg # for max
        if gap > 1e-3:
            print("ERROR IN SUPPORT SET CALCULATION")
        print("gap of {:.3f}".format(gap))

        #actual_supports = []
        #for j in tqdm(range(args["num_samples"])):
        #    #print("Testing sample {} of {}".format(j+1,args["num_samples"]))
        #    samples_new = remove_sample(j, samples)
        #    test_res, test_pol, _, _ = solve_subgrad(samples_new, model, max_iters=100, quiet=True)
        #    if model.opt == "max":
        #        gap = abs(res_sg - test_res)
        #    else:
        #        gap = abs(res_sg - test_res)
        #    logging.info("Calculated a difference of {} with sample {} removed".format(gap,j+1))
        #    if gap >= 1e-4:
        #        actual_supports.append(j)
        #        if j not in active_sg:
        #            print("ERROR, found an empirical SC not in a posteriori support set")
        #            num_errors += 1
        #print("Empirical supports: " + str([elem+1 for elem in actual_supports]))
        #if len(actual_supports) > model.max_supports:
        #    print("ERROR, a priori max sc was wrong!")
        #    num_underestimates += 1
        #for sc in active_sg:
        #    if sc not in actual_supports:
        #        num_over += 1
        #        print("Expected sample {} to be support but it wasn't".format(sc))
    print("Found {} samples which were expected to be of support that were not".format(num_over))
    print("Found {} samples which were not identified as support samples".format(num_errors))
    print("A priori underestimated number of supports {} times".format(num_underestimates))

def remove_sample(pos, samples):
    new_samples = copy.copy(samples)
    new_samples.pop(pos)
    return new_samples
