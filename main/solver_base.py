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
        self.save_plots = args["save_figs"]

    def solve(self, samples, model):
        start = time.perf_counter()
        self.opt, self.opt_pol, self.supps, self.info = self.optimiser.solve(samples, model)
        if model.switch_res:
            self.opt = 1-self.opt

        if self.supps is not None:
            self.risk = self.get_risk(self.beta, len(samples), len(self.supps))
        else:
            self.risk = 1
        self.runtime = time.perf_counter()-start
        if self.run_MC:
            self.emp_risk = MC_sampler(model, self.MC_samples, self.opt, self.opt_pol)
        if self.run_MC_pert:
            self.emp_pert_risk = MC_perturbed(model, self.MC_samples, self.opt, self.opt_pol)

    def plot_hist(self, opt_sat=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        if opt_sat == None:
            opt_sat = self.opt 
            self.info["hist"].pop(-1)
            ax.set_ylabel("Distance from final satisfaction probability")
        else:
            ax.set_ylabel("Distance from optimal satisfaction probability")
        if opt_sat > self.info["hist"][0]:
            res_plot = [opt_sat - i for i in self.info["hist"]]
        else:
            res_plot = [i-opt_sat for i in self.info["hist"]]
        ax.semilogy(res_plot)
        ax.set_xlabel("Iteration")


        if self.save_plots:
            fname = self.save_plots + 'dist_fig'
            plt.savefig(fname + ".png", bbox_inches="tight")
            plt.savefig(fname + ".pdf", bbox_inches="tight")
        else:
            plt.show()

        fig2, ax2 = plt.subplots()
        ax2.semilogy(self.info["hist"])
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Satisfaction probability")

        if self.save_plots:
           fname = self.save_plots + 'prob_fig'
           plt.savefig(fname + ".png", bbox_inches="tight")
           plt.savefig(fname + ".pdf", bbox_inches="tight")
        else:
            plt.show()

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
