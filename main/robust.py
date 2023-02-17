import numpy as np
import cvxpy as cp
import Markov.writer as writer
from PAC.funcs import *
import itertools

def calc_probs_exhaustive(model, N):
    """
    Doesn't work! Policy needs to be non-deterministic
    """
    min_prob = 1
    discarded = 0
    pols = list(itertools.product(*model.Enabled_actions)) 
    
    probs = [[] for i in pols]
    for i in range(N):
        sample = model.sample_MDP()
        
        for i, pol in enumerate(pols):
            sampled_MC = sample.fix_pol(pol)
        #IO = writer.PRISM_io(sample)
            IO = writer.stormpy_io(sampled_MC)
            IO.write()
        #IO.solve_PRISM()
            res, all_res = IO.solve()
            probs[i] += res
    min_probs = [min(elem) for elem in probs]
    max_min = max(min_probs)
    pol_ind = min_probs.index(max_min)
    pol = pols[pol_ind]
    return probs[pol_ind], pol

def calc_probs_policy_iteration(model, N, max_iters=1000):
    sampled_trans_probs = []
    for i in range(N):
        sample = model.sample_MDP()
        sampled_trans_probs.append(sample.Transition_probs)
   
    import pdb; pdb.set_trace()
    num_states = len(model.States)
    num_acts = len(model.Actions)

    probs = cp.Parameter((num_states,N))
    prob_init = np.zeros((num_states,N))
    for reached in model.Labelled_states[model.Labels.index("reached")]:
        prob_init[reached, :] = 1.0
    probs.value = prob_init 
    
    pi = cp.Variable((num_states, num_acts))
    new_probs = cp.Variable((num_states, N))
    worst_case = cp.Variable(num_states)
    objective = cp.Maximize(np.ones((num_states,1)).T@worst_case)
    A = np.ones((num_acts, 1))
    b = np.ones((num_states, 1))
    for i in range(max_iters):

        #something is not quite right below?? Probabilities not being "pulled up" to correct values

        constraints = [new_probs <= 1, new_probs >= 0, pi@A == b, pi >= 0]
        probs_s_a = [
                        [
                            [
                                sum([sampled_trans_probs[k][s_num][a_num][s_prime_num]\
                                        *probs[s_prime,k] 
                                     for s_prime_num, s_prime in enumerate(model.trans_ids[s_num][a_num])]) 
                                for a_num, a_id in enumerate(model.Enabled_actions[s_id])] 
                            for s_num, s_id in enumerate(model.States)] 
                        for k in range(N)
                    ]

        for k in range(N):
            constraints += [worst_case[s] <= new_probs[s,k] for s in model.States]
            constraints += [new_probs[s,k] <= 
                         sum([pi[s,a]*probs_s_a[k][s_num][a_num] 
                              for a_num, a in enumerate(model.Enabled_actions[s_num])]) 
                             for s_num, s in enumerate(model.States)]
        program = cp.Problem(objective, constraints)
        result = program.solve()
        probs.value = new_probs.value
        print(pi.value)
        print(worst_case.value)
        import pdb; pdb.set_trace()

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

    etas = x_s.value[1:]
    tau = x_s.value[0]
    


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

def optimise(rho, probs):
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
    model = args["model"]
    probs, pol = calc_probs_policy_iteration(model, args["num_samples"])
    min_prob, discarded = discard(args["lambda"], probs)
    tau, etas = optimise(args["rho"], probs)
    [epsL, epsU] = calc_eps_risk_complexity(1-args["beta"], args["num_samples"], np.sum(etas>=0))

    print("Using results from risk and complexity, new sample will satisfy formula with lower bound {:.3f}, with a violation probability in the interval [{:.3f}, {:.3f}] with confidence {:.3f}".format(tau, epsL, epsU, args["beta"]))
    if args["lambda"] < 1:
        thresh = calc_eta_discard(args["beta"], args["num_samples"], discarded)
        print("Discarded {} samples".format(discarded))
    else:
        thresh = calc_eta_var_thresh(args["beta"], args["num_samples"])

    print(("Probability of new sample satisfying formula with probability at least {:.3f}"+
               " is found to be {:.3f}, with confidence {:.3f}.").format(min_prob, thresh, args["beta"]))

    if args["MC"]:
        out, inn = MC_sampler(model, args["MC_runs"], args["MC_samples"], min_prob, thresh, pol) 
        print("Empirical violation rate is found to be (on average) {:.3f}, with confidence {:.3f}".format(inn,out))

if __name__=="__main__":
    main()

