import Models.test as mod
import numpy as np
import Markov.writer as writer
from UI.get_args import run as get_args
from PAC.funcs import *

def main():
    args = get_args()
    discarding = args["eta"] is not None
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
            if not discarding:
                if res[0] < min_prob:
                    min_prob = res[0]
            else:
                if res[0] < args["eta"]:
                    discarded += 1
        print(discarded)
        if discarding:
            min_prob = args["eta"]
            thresh = calc_eta_discard(args["beta"], args["num_samples"], discarded)
        else:
            thresh = calc_eta_var_thresh(args["beta"], args["num_samples"])

        print(("Probability of new sample satisfying formula with probability {}"+
              " is found to be {}, with confidence {}.").format(min_prob, thresh, args["beta"]))

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

