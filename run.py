import Models.test as test
import numpy as np
import cvxpy as cp
import Markov.writer as writer
from UI.get_args import run as get_args
from PAC.funcs import *
from main.individual import run_all
from main.robust import run_all as robust_run

def main():
    args = get_args()
    if args["model"] == "test":
        model = test.get_model()
    elif args["model"] == "test2":
        model = test.get_model_2()
    elif args["model"] == "test3":
        model = test.get_model_3()
    else:
        raise NotImplementedError
    run_all(model, args)
    #robust_run(model, args)

if __name__=="__main__":
    main()

