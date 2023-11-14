from UI.get_args import run as get_args

from main.solver_base import solver
from main.solvers import *

from main.MNE_algos import PNS_algo, FSP_algo
import datetime
import sys

def main():
    args = get_args()
    samples = [get_samples(args)[0]]
     
    if args["file_write"]:
        sys.stdout = open(args["file_write"],'wt')

    model = args["model"]
    solvers = []
    
    solvers.append(solver(subgrad, args))
    solvers.append(solver(bellman, args)) 
    
    for sol in solvers:
        if len(model.States) < 100:
            sol.optimiser.parallel_grad =False
        if len(samples) < 50:
            sol.parallel_test = False
        sol.solve(samples, model)
        sol.output() 
    import pdb; pdb.set_trace()

if __name__=="__main__":
    main()

