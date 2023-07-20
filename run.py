import Models.test as test
import numpy as np
import cvxpy as cp
import Markov.writer as writer
from UI.get_args import run as get_args
from PAC.funcs import *
from main.individual import run_all
from main.solvers import run_all as robust_run
from main.sampler import get_samples
import logging

def main():
    args = get_args()
    samples = get_samples(args)
    run_all(args, samples)
    robust_run(args, samples)

if __name__=="__main__":
    main()

