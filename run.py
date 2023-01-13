import Models.test as mod
import numpy as np
import cvxpy as cp
import Markov.writer as writer
from UI.get_args import run as get_args
from PAC.funcs import *

def main():
    args = get_args()
    discarding = args["lambda"] is not None
    optimising = args["rho"] is not None
    if args["test"]:
        test = mod.get_model()
        probs = []
        min_prob = 1
        discarded = 0
        for i in range(args["num_samples"]):
            sample = test.sample_MDP()
            
            #IO = writer.PRISM_io(sample)
            IO = writer.stormpy_io(sample)
            IO.write()
            #IO.solve_PRISM()
            res, all_res = IO.solve()
            probs += res
            p_val = res[0]
            if discarding:
                if res[0] < args["lambda"]:
                    discarded += 1
                    p_val = min_prob
            if p_val < min_prob:
                min_prob = p_val
        if optimising:
            N = args["num_samples"]
            x_s = cp.Variable(1+N)
            c = np.ones((N+1,1))*args["rho"]
            c[0] = 1

            b = np.array([0]+probs)
            A = np.eye(N+1)
            A[:, 0] = -1
            A = -A

            import pdb; pdb.set_trace()
            objective = cp.Minimize(c.T@x_s)
            constraints = [A@x_s >= b]
            prob = cp.Problem(objective, constraints)
            result = prob.solve()
            print(x.value)
            import pdb; pdb.set_trace()
        print(discarded)
        if discarding:
            thresh = calc_eta_discard(args["beta"], args["num_samples"], discarded)
        else:
            thresh = calc_eta_var_thresh(args["beta"], args["num_samples"])

        print(("Probability of new sample satisfying formula with probability {:.3f}"+
               " is found to be {:.3f}, with confidence {:.3f}.").format(min_prob, thresh, args["beta"]))

        #opt_pol, rew = IO.read()
        #print(min_prob)
        #pol = np.zeros((6),dtype='int')
        #
        #chain = sample.fix_pol(pol)
        #
        #IO = writer.stormpy_io(chain)
        #IO.write()
        ##IO.solve_PRISM()
        #res, _ = IO.solve()

if __name__=="__main__":
    main()

