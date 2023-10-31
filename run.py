from UI.get_args import run as get_args

from main.solver_base import solver
from main.solvers import *

from main.MNE_algos import PNS_algo, FSP_algo
import datetime
import sys

def main():
    args = get_args()
    samples = get_samples(args)
    
    model = args["model"]

    sub = solver(subgrad, args)
    sub.solve(samples, model)
    sub.output()

    det_solver = solver(det, args)
    det_solver.solve(samples, model)
    det_solver.output()
    
    PNS_solver = solver(MNE, args, [PNS_algo])
    PNS_solver.solve(samples, model)
    PNS_solver.output()
    
    FSP_solver = solver(MNE, args, [FSP_algo])
    FSP_solver.solve(samples, model)
    FSP_solver.output()

    iMDP_solver = solver(interval, args)
    iMDP_solver.solve(samples, model)
    iMDP_solver.output()
    
    thom_discard_solver = solver(thom_discard, args)
    thom_discard_solver.solve(samples, model)
    thom_discard_solver.output()
    
    thom_relax_solver = solver(thom_relax, args)
    thom_relax_solver.solve(samples, model)
    thom_relax_solver.output()

if __name__=="__main__":
    main()

