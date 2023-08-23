import Markov.writer as writer
import time
from PAC.funcs import *

def solve_imdp(model, samples):
    iMDP = model.build_imdp(samples)
    supports = len(iMDP.supports)
    IO = writer.PRISM_io(iMDP)
    IO.write()
    res, all_res, pol = IO.solve()
    return res, pol, supports

def run(args, samples):
    start = time.perf_counter()
    N = args["num_samples"]
    model = args["model"]
    print("Finding iMDP probability\n----------------")
    if N > 2*model.max_supports:
        [a_priori_eps_L, a_priori_eps_U] = \
            calc_eps_risk_complexity(args["beta"], N, model.max_supports*2)
    else:
        [a_priori_eps_L, a_priori_eps_U] = \
            calc_eps_risk_complexity(args["beta"], N, N)


    print("A priori, violation probability is in the range [{:.3f}, {:.3f}], with confidence {:.3f}"
                .format(a_priori_eps_L, a_priori_eps_U, args["beta"]))
    prob, pol, supports = solve_imdp(model, samples)
    [a_post_eps_L, a_post_eps_U] = \
        calc_eps_risk_complexity(args["beta"], N, supports)
    print("Time for iMDP solving: {:.3f}s".format(time.perf_counter()-start))

    print("Using iMDP found " + str(supports) + " possible support constraints")

    print("Hence, a posteriori, violation probability is in the range [{:.3f}, {:.3f}], with confidence {:.3f}"
                .format(a_post_eps_L, a_post_eps_U, args["beta"]))
    
    print("Optimal satisfaction probability is found to be {:.3f}".format(prob[0]))
        
    if args["MC"]:
        emp_violation = MC_sampler(model, args["MC_samples"], prob[0], pol) 
        print("Empirical violation rate is found to be {:.3f}".format(emp_violation))
    if args["MC_pert"]:
        pert_violation = MC_perturbed(model, args["MC_samples"], prob[0], pol) 
        print("Noisy violation rate is found to be {:.3f}".format(pert_violation))
