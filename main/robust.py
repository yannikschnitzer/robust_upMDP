import numpy as np
import cvxpy as cp
import Markov.writer as writer
from PAC.funcs import *
import itertools

def calc_probs_exhaustive(model, N):
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

def run_all(model, args):
    probs, pol = calc_probs_exhaustive(model, args["num_samples"])
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

