import numpy as np
import cvxpy as cp
import Markov.writer as writer
from PAC.funcs import *
import itertools
import logging
from tqdm import tqdm
import time

def gen_samples(model, N):
    sampled_trans_probs = []
    for i in range(N):
        sample = model.sample_MDP()
        sampled_trans_probs.append(sample.Transition_probs)
    return sampled_trans_probs

def calc_reach_sets(model):
    backward_reach = [[] for s in model.States]
    for state in model.States:
        successors = set()
        for elem in model.trans_ids[state]:
            successors.update(elem)
        for succ in successors:
            backward_reach[succ].append(state)
    return backward_reach

def calc_max_path(backward_reach):

    # not implemented yet
    return len(backward_reach)

def calc_probs_policy_iteration(model, samples, max_iters=10000, tol=1e-5):
  
    # test for test 2, should get optimal policy 0.5, 0.5 and value 0.65
    #samples = [samples[0], samples[1]]
    #samples[0][1][0] = [0.7, 0.3]
    #samples[0][2][0] = [0.6,0.4]
    #samples[1][1][0] = [0.5, 0.5]
    #samples[1][2][0] = [0.8, 0.2]
    N = len(samples)
    
    back_set = calc_reach_sets(model)
    num_states = len(model.States)
    num_acts = len(model.Actions)
    
    states_to_update = set()
    probs = cp.Parameter((num_states,N))
    prob_init = np.zeros((num_states,N))
    for reached in model.Labelled_states[model.Labels.index("reached")]:
        prob_init[reached, :] = 1.0
        states_to_update.update(back_set[reached])
    probs.value = prob_init 
    
    pol = np.ones((num_states, num_acts))/num_acts
    pi = cp.Variable(num_acts)
    new_prob = cp.Variable(N)
    worst_prob = cp.Variable(1)
    objective = cp.Maximize(worst_prob)
    converged=False
    total_time = 0
    for i in range(max_iters):
        tic = time.perf_counter()
        next_states_to_update = set()
        prob_updates = {}
        for s in tqdm(states_to_update):
            constraints = [new_prob <= 1, new_prob >= 0, np.ones(num_acts)@pi == 1, pi >= 0]
            for k in range(N):
                constraints += [worst_prob <= new_prob[k]]
                constraints += [new_prob[k] == 
                             sum([pi[a]*sum([samples[k][s][a_num][s_prime_num]*probs[s_prime,k] for s_prime_num, s_prime in enumerate(model.trans_ids[s][a_num])])  
                                  for a_num, a in enumerate(model.Enabled_actions[s])]) 
                                 ]
            logging.debug("problem construction complete, moving on to solving")
            program = cp.Problem(objective, constraints)
            result = program.solve(ignore_dpp=True)
            changed = np.any(abs(probs.value[s]-new_prob.value)>=tol)
            if changed:
                next_states_to_update.update(back_set[s])
            # Do I have to wait until the end of this iteration to update prob values??
            #prob_updates[s] = new_prob.value
            probs.value[s] = new_prob.value
            pol[s] = pi.value
        states_to_update = next_states_to_update
        #for s in prob_updates:
        #    probs.value[s] = prob_updates[s]
        if len(states_to_update) == 0:
            converged=True
            break
        toc = time.perf_counter()
        total_time += toc-tic
        logging.info("iteration {} completed in {:.3f}s".format(i, toc-tic))
        logging.info("Current worst case probabilities are {}".format(np.min(probs.value, axis=1)))
    logging.info("Entire optimization finished in {:.3f}s".format(total_time))
    import pdb; pdb.set_trace()

    # sometimes we find an additional support sample
    num_supports = 0
    support_samples = set()
    for s in range(num_states):
        max_sc = len(model.Enabled_actions[s])
        found_sc = np.argwhere(probs.value[s] <= worst_case.value[s]+tol/2)
        if found_sc.size <= max_sc:
            support_samples.update([int(elem) for elem in found_sc])
    num_supports = len(support_samples)
    if converged:
        return probs.value[model.Init_state], pi.value, support_samples, num_supports
    else:
        return -1, -1, -1, -1

def discard(lambda_val, probs):
    min_prob = 1
    discarded=0
    undiscarded = [p_val for p_val in probs if p_val > lambda_val]

    if len(undiscarded)>0:
        min_prob = min(undiscarded)
    else:
        min_prob = lambda_val
    discarded = len(probs)-len(undiscarded)

    return min_prob, discarded

def with_relaxation(rho, probs):
    N = len(probs)
    x_s = cp.Variable(1+N)
    c = -np.ones((N+1,1))*rho
    c[0] = 1

    b = np.array([0]+probs)
    A = np.eye(N+1)
    A[:, 0] = -1
    A[0,0] = 1
    A = -A

    objective = cp.Maximize(c.T@x_s)
    constraints = [A@x_s <= b, x_s >= 0]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    etas = x_s.value[1:]
    tau = x_s.value[0]
    
    return tau, etas

def run_all(args):
    print("Running code for robust optimal policy \n --------------------")
    model = args["model"]
    
    a_priori_max_supports = sum([len(acts) for acts in model.Enabled_actions])

    a_priori_eps = calc_eps(args["beta"], args["num_samples"], a_priori_max_supports)
    
    print("A priori upper bound on number of support constraints is " + str(a_priori_max_supports))

    print("A priori bound on violation probability is {:.3f} with confidence {:.3f}".format(a_priori_eps, args["beta"]))

    samples = gen_samples(model, args["num_samples"])

    probs, pol, supports, a_post_support_num  = calc_probs_policy_iteration(model, samples)
    
    [a_post_eps_L, a_post_eps_U] = \
        calc_eps_risk_complexity(args["beta"], args["num_samples"], a_post_support_num)
    
    print("A posteriori, found " + str(a_post_support_num) + " support constraints")

    print("A posteriori, violation probability is in the range [{:.3f}, {:.3f}], with confidence {:.3f}"
          .format(a_post_eps_L, a_post_eps_U, args["beta"]))

    min_prob = min(probs)

    print("Minimum satisfaction probability is found to be {:.3f}".format(min_prob))

    if pol.size < 50:
        print("Calculated robust policy is:")
        print(pol)

    thresh = a_priori_eps

    if args["MC"]:
        emp_violation = MC_sampler(model, args["MC_samples"], min_prob, thresh, pol) 
        print("Empirical violation rate is found to be {:.3f}".format(emp_violation))
    print("\n\n")

def test_support_num(args):
    model = args["model"]
    for i in range(100):
        samples = gen_samples(model, args["num_samples"])

        probs, pol, supports, a_post_support_num  = calc_probs_policy_iteration(model, samples)
        
        print("Calculated supports: " + str(supports))
        actual_supports = []
        for i, sample in enumerate(tqdm(samples)):
                samples.pop(i)
                test_probs, test_pol, _, _ = calc_probs_policy_iteration(model, samples)
                if abs(min(probs) - min(test_probs)) >= 0.01:
                    actual_supports.append(i)
                samples.insert(i, sample)
        print("Emprical supports: " + str(actual_supports))
        if len(actual_supports) > a_post_support_num:
            print("ERROR")
            import pdb; pdb.set_trace()

if __name__=="__main__":
    main()

