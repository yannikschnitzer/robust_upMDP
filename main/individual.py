import numpy as np
import cvxpy as cp
import Markov.writer as writer
from PAC.funcs import *
import time

def calc_probs(model, samples):
    probs = []
    min_prob = 1
    discarded = 0
    for sample in samples:
        
        test_MDP = model.fix_params(sample)
        #IO = writer.PRISM_io(sample)
        IO = writer.stormpy_io(test_MDP)
        IO.write()
        #IO.solve_PRISM()
        res, all_res, _ = IO.solve()
        probs += res
    return probs


def discard(lambda_val, probs, opt):
    if lambda_val is None:
        if opt == "max":
            opt_prob = min(probs)
        else:
            opt_prob = max(probs)
        discarded = 0
    else:
        if opt == "max":
            undiscarded = [p_val for p_val in probs if p_val >= lambda_val]
            if len(undiscarded)>0:
                opt_prob = min(undiscarded)
            else:
                opt_prob = 0
        else:
            undiscarded = [p_val for p_val in probs if p_val <= lambda_val]
            if len(undiscarded)>0:
                opt_prob = max(undiscarded)
            else:
                opt_prob = 1

        discarded = len(probs)-len(undiscarded)
    return opt_prob, discarded

def optimise(rho, probs, opt):
    N = len(probs)
    x_s = cp.Variable(1+N)
    c = -np.ones((N+1,1))*rho
    if opt == "max":
        c[0] = 1
        b = np.array([0]+probs)
        A = np.eye(N+1)
        A[:, 0] = -1
        A[0,0] = 1
        A = -A
        constraints = [A@x_s <= b, x_s >= 0]
    else:
        c[0] = -1
        b = np.array([-1]+probs)
        A = np.eye(N+1)
        A[:, 0] = 1
        A[0,0] = -1
        #A = -A
        constraints = [A@x_s >= b, x_s >= 0]

    objective = cp.Maximize(c.T@x_s)


    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    etas = x_s.value[1:]
    tau = x_s.value[0]
    
    return tau, etas

def run_all(args, samples):
    start = time.perf_counter()
    model = args["model"]
    print("Running code for individual optimal policies \n --------------------")
    probs = calc_probs(model, samples)
    min_prob, discarded = discard(args["lambda"], probs, model.opt)
    runtime = time.perf_counter()-start
    print("Time for finding probabilities and discarding: {:.3f}s".format(runtime))

    #tau, etas = optimise(args["rho"], probs, model.opt)
    #[epsL, epsU] = calc_eps_risk_complexity(args["beta"], args["num_samples"], np.sum(etas>=0))

    #print("Using results from risk and complexity, new sample will satisfy formula with bound {:.3f}, with a violation probability in the interval [{:.3f}, {:.3f}] with confidence {:.3f}".format(tau, epsL, epsU, args["beta"]))
    if args["lambda"] is not None:
        thresh = calc_eta_discard(args["beta"], args["num_samples"], discarded)
        print("Discarded {} samples".format(discarded))
    else:
        thresh = calc_eta_var_thresh(args["beta"], args["num_samples"])

    print(("Upper bound on violation probability for formula with probability bounded by {:.3f}"+
               " is found to be {:.3f}, with confidence {:.3f}.").format(min_prob, thresh, args["beta"]))

    if args["MC"]:
        emp_violation = MC_sampler(model, args["MC_samples"], min_prob) 
        print("Empirical violation rate is found to be {:.3f}".format(emp_violation))
    
    if args["MC_pert"]:
        emp_violation = MC_perturbed(model, args["MC_samples"], min_prob) 
        print("Noisy violation rate is found to be {:.3f}".format(emp_violation))
    print("\n\n")
    return runtime

if __name__=="__main__":
    main()

