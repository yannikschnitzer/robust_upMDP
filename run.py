from UI.get_args import run as get_args

from main.solver_base import solver
from main.solvers import *

from main.MNE_algos import PNS_algo, FSP_algo
import datetime
import sys

def main():
    args = get_args()
    samples = get_samples(args)
    
    print("Generated Samples: ")
    for sample in samples:
        print(sample)
    print("HER --------------")
    if args["file_write"]:
        sys.stdout = open(args["file_write"],'wt')

    print("HER --------------")
    model = args["model"]
    solvers = []
    
    print("Runnging Subgrad Solver-----------------------------------")
    solvers.append(solver(subgrad, args))
    # solvers.append(solver(interval, args))
    # solvers.append(solver(thom_discard, args))
    
    if not args["sg_only"]:
        solvers.append(solver(det, args))
        solvers.append(solver(MNE, args, [PNS_algo]))
        solvers.append(solver(MNE, args, [FSP_algo]))
        solvers.append(solver(thom_relax, args))
    
    for sol in solvers:
        if len(model.States) < 100:
            sol.optimiser.parallel_grad =False
        if len(samples) < 50:
            sol.parallel_test = False
        print("Here------------------------------------------")
        sol.solve(samples, model)
        print("About to output-----------------------------------")
        sol.output() 
    
    if args["output_figs"] or args["save_figs"]:
        if not args["sg_only"]:
            solvers[0].plot_hist(solvers[4].opt)
        else:
            solvers[0].plot_hist()


if __name__=="__main__":
    main()

