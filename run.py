import Models.test as test
import numpy as np
import cvxpy as cp
import Markov.writer as writer
from UI.get_args import run as get_args
from PAC.funcs import *
from main.individual import run_all
from main.FSP import run_all as robust_run
import logging

def main():
    args = get_args()
    #run_all(args)
    robust_run(args)

if __name__=="__main__":
    main()

