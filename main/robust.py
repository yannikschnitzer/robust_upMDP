import numpy as np
import cvxpy as cp
import Markov.writer as writer
from PAC.funcs import *
import itertools
from tqdm import tqdm

def gen_samples(model, N):
    sampled_trans_probs = []
    for i in range(N):
        sample = model.sample_MDP()
        sampled_trans_probs.append(sample.Transition_probs)
    return sampled_trans_probs

def calc_probs_policy_iteration(model, samples, max_iters=1000, tol=1e-5):
  
    N = len(samples)

    num_states = len(model.States)
    num_acts = len(model.Actions)

    probs = cp.Parameter((num_states,N))
    prob_init = np.zeros((num_states,N))
    for reached in model.Labelled_states[model.Labels.index("reached")]:
        prob_init[reached, :] = 1.0
    probs.value = prob_init 
  
    # test for test 2, should get optimal policy 0.5, 0.5 and value 0.65
    #samples = [samples[0], samples[1]]
    #samples[0][1][0] = [0.7, 0.3]
    #samples[0][2][0] = [0.6,0.4]
    #samples[1][1][0] = [0.5, 0.5]
    #samples[1][2][0] = [0.8, 0.2]
    
    pi = cp.Variable((num_states, num_acts))
    new_probs = cp.Variable((num_states, N))
    worst_case = cp.Variable(num_states)
    objective = cp.Maximize(np.ones((num_states,1)).T@worst_case)
    A = np.ones((num_acts, 1))
    b = np.ones((num_states, 1))
    old_wc = np.ones(num_states)
    converged=False
    for i in range(max_iters):
        constraints = [new_probs <= 1, new_probs >= 0, pi@A == b, pi >= 0]
        # might be more efficient to only check states that will change 
        # (i.e. with a transition to a state that has a changed value for probs)
        for k in tqdm(range(N)):
            constraints += [worst_case[s] <= new_probs[s,k] for s in model.States]
            constraints += [new_probs[s,k] == 
                         sum([pi[s,a]*sum([samples[k][s_num][a_num][s_prime_num]*probs[s_prime,k] for s_prime_num, s_prime in enumerate(model.trans_ids[s_num][a_num])])  
                              for a_num, a in enumerate(model.Enabled_actions[s_num])]) 
                             for s_num, s in enumerate(model.States)]
        program = cp.Problem(objective, constraints)
        result = program.solve()
        diff = np.linalg.norm(probs.value - new_probs.value, ord=np.inf)
        print("Infinity norm change at iteration {} is {:.3f}".format(i, diff))
        if diff <= tol:
            converged=True
            break
        old_wc = worst_case.value
        probs.value = new_probs.value
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

