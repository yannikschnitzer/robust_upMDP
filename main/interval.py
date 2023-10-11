import Markov.writer as writer
import logging
import time
from PAC.funcs import *
from opts import prism_folder  

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

def solve_imdp(model, samples):
    iMDP = model.build_imdp(samples)
    supports = len(iMDP.supports)
    IO = writer.PRISM_io(iMDP)
    IO.write()
    res, all_res, pol = IO.solve(prism_folder=prism_folder)
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
    runtime = time.perf_counter()-start
    print("Time for iMDP solving: {:.3f}s".format(runtime))

    wc, true_p, _ = test_pol(model, samples, pol)

    print("Using iMDP found " + str(supports) + " possible support constraints")

    print("Hence, a posteriori, violation probability is in the range [{:.3f}, {:.3f}], with confidence {:.3f}"
                .format(a_post_eps_L, a_post_eps_U, args["beta"]))
    
    print("Optimal satisfaction probability for iMDP is found to be {:.3f}".format(prob[0]))
    print("Optimal satisfaction probability for iMDP policy on upMDP is {:.3f}".format(wc))

    if args["MC"]:
        emp_violation = MC_sampler(model, args["MC_samples"], wc, pol) 
        print("Empirical violation rate is found to be {:.3f}".format(emp_violation))
    if args["MC_pert"]:
        pert_violation = MC_perturbed(model, args["MC_samples"], wc, pol) 
        print("Noisy violation rate is found to be {:.3f}".format(pert_violation))

    return runtime
