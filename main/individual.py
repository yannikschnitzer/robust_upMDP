import numpy as np
import cvxpy as cp
import Markov.writer as writer
from PAC.funcs import *

def calc_probs(model, N):
    probs = []
    min_prob = 1
    discarded = 0
    for i in range(N):
        sample = model.sample_MDP()
        
        #IO = writer.PRISM_io(sample)
        IO = writer.stormpy_io(sample)
        IO.write()
        #IO.solve_PRISM()
        res, all_res = IO.solve()
        probs += res
    return probs


def discard(lambda_val, probs):
    min_prob = 1
    discarded=0
    undiscarded = [p_val for p_val in probs if p_val >= lambda_val]
    
    if len(undiscarded)>0:
        min_prob = min(undiscarded)
    else:
        min_prob = 0
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
    print("Running code for individual optimal policies \n --------------------")
    probs = calc_probs(model, args["num_samples"])
    min_prob, discarded = discard(args["lambda"], probs)
    tau, etas = optimise(args["rho"], probs)
    [epsL, epsU] = calc_eps_risk_complexity(1-args["beta"], args["num_samples"], np.sum(etas>=0))

    print("Using results from risk and complexity, new sample will satisfy formula with lower bound {:.3f}, with a violation probability in the interval [{:.3f}, {:.3f}] with confidence {:.3f}".format(tau, epsL, epsU, args["beta"]))
    if args["lambda"] > 0:
        thresh = calc_eta_discard(args["beta"], args["num_samples"], discarded)
        print("Discarded {} samples".format(discarded))
    else:
        thresh = calc_eta_var_thresh(args["beta"], args["num_samples"])

    print(("Upper bound on violation probability for formula with probability at least {:.3f}"+
               " is found to be {:.3f}, with confidence {:.3f}.").format(min_prob, thresh, args["beta"]))

    if args["MC"]:
        out, min_inn, inn, max_inn = MC_sampler(model, args["MC_samples"], args["MC_runs"], min_prob, thresh, None) 
        print("Empirical violation rate is found to be between {:.3f} and {:.3f} with mean {:.3f}, with confidence {:.3f}".format(min_inn, max_inn, inn, out))
    print("\n\n")

if __name__=="__main__":
    main()

