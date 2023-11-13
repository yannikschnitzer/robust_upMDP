from PAC.funcs import MC_sampler, MC_perturbed 
import time

class solver:
    """
    Base class for solvers
    """
    def __init__(self, solver, args, extra_opt_args=[]):
        #if extra_opt_args is not None:
        self.optimiser = solver(args, *extra_opt_args)

        self.get_risk = self.optimiser.risk_func
        
        self.opt = None
        self.opt_pol = None
        self.supps = None
        self.info = None

        self.beta = args["beta"]
        self.run_MC = args["MC"]
        self.run_MC_pert = args["MC_pert"]
        self.MC_samples = args["MC_samples"]

    def solve(self, samples, model):
        start = time.perf_counter()
        self.opt, self.opt_pol, self.supps, self.info = self.optimiser.solve(samples, model)
        if self.supps is not None:
            self.risk = self.get_risk(self.beta, len(samples), len(self.supps))
        else:
            self.risk = 1
        self.runtime = time.perf_counter()-start
        if self.run_MC:
            self.emp_risk = MC_sampler(model, self.MC_samples, self.opt, self.opt_pol)
        if self.run_MC_pert:
            self.emp_pert_risk = MC_perturbed(model, self.MC_samples, self.opt, self.opt_pol)

    def output(self):
        print("Solving took {:.2f}s".format(self.runtime))

        print("Found " + str(len(self.supps)) + " active constraints a posteriori")

        print("Hence, a posteriori, violation probability is bounded by {:.3f}, with confidence {:.3f}"
                .format(self.risk, self.beta))

        print("Optimal satisfaction probability is found to be {:.3f}".format(self.opt))

        if self.run_MC:
            print("Empirical violation rate with {} samples found to be: {:.3f}".format(self.MC_samples, self.emp_risk))
        if self.run_MC_pert:
            print("Violation rate for perturbed problem with {} samples found to be: {:.3f}".format(self.MC_samples, self.emp_pert_risk))
