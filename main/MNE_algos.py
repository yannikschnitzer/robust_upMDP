import numpy as np
import cvxpy as cp
import time
from tqdm import tqdm
import itertools

def check_timeout(start, max_t):
    return time.perf_counter() - start > max_t

def PNS_algo(payoffs, max_t, itts):
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
        if check_timeout(start, max_t):
            return None, None
        supps_pols = list(itertools.combinations(range(strat_lengths[0]), x_1))
        for S_pol in supps_pols:
            if check_timeout(start, max_t):
                return None, None
            redux_payoffs = payoffs[S_pol, :]
            Adv_act_prime = []
            for i, elem in enumerate(redux_payoffs.T):
                if check_timeout(start, max_t):
                    return None, None
                if not np.any(np.all(elem.T[:,np.newaxis] > redux_payoffs, axis=0)):
                    # elem not dominated
                    Adv_act_prime.append(i)
            redux_redux_payoffs = redux_payoffs[:, Adv_act_prime]
            conditional_dommed_exists = False
            for i, elem in enumerate(redux_redux_payoffs):
                if check_timeout(start, max_t):
                    return None, None
                if np.any(np.all(elem[np.newaxis, :] < redux_redux_payoffs, axis=1)):
                    conditional_dommed_exists = True
            if not conditional_dommed_exists:
                supps_adv = list(itertools.combinations(Adv_act_prime, x_2))
                for S_adv in supps_adv:
                    if check_timeout(start, max_t):
                        return None, None
                    r_r_r_payoffs = redux_payoffs[:, S_adv]
                    conditional_dommed_exists_2 = False
                    for i, elem in enumerate(r_r_r_payoffs):
                        if check_timeout(start, max_t):
                            return None, None
                        if np.any(np.all(elem[np.newaxis, :] < r_r_r_payoffs, axis=1)):
                            conditional_dommed_exists_2 = True
                    if not conditional_dommed_exists_2:
                        pol, val = Feas_prog(payoffs, S_pol, S_adv)
                        if pol is not None:
                            return pol, val
    return None, None




def FSP_algo(payoffs, max_time, max_iters = 100000):
    start = time.perf_counter()
    update_parallel = True
    
    pol_dist = np.ones((1,payoffs.shape[0]))/payoffs.shape[0]
    sample_dist = np.ones((payoffs.shape[1],1))/payoffs.shape[1]

    sum_step = 0
    print("---------------------\nStarting FSP")
    for i in tqdm(range(max_iters)):
        if check_timeout(start, max_time):
            break     
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
            pol_dist[0, np.argmax(pol_vec)] += new
            sample_dist[np.argmin(sample_vec), 0] += new
    res = np.min(pol_dist@payoffs)
    return pol_dist, res
    #info = {"pols": pols, "all":(pol_dist@payoffs).flatten(), "ids":rel_samples}
    #return res, pol_dist.flatten(), np.sum(sample_dist >= 1e-4), info

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

