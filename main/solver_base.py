
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

    def solve(self, samples, model):
        self.opt, self.opt_pol, self.supps, self.info = self.optimiser.solve(samples, model)
        self.risk = self.get_risk(self.beta, len(samples), len(self.supps))

    def output(self):
        print("Found " + str(len(self.supps)) + " active constraints a posteriori")

        print("Hence, a posteriori, violation probability is bounded by {:.3f}, with confidence {:.3f}"
                .format(self.risk, self.beta))

        print("Optimal satisfaction probability is found to be {:.3f}".format(self.opt))

