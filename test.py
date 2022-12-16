import Models.test as mod
import numpy as np
import Markov.writer as writer
from UI.get_args import run as get_args

def main():
    args = get_args()
    test = mod.get_model()
    sample = test.sample_MDP()
    
    #IO = writer.PRISM_io(sample)
    IO = writer.stormpy_io(sample)
    IO.write()
    #IO.solve_PRISM()
    res = IO.solve()
    
    #opt_pol, rew = IO.read()
    
    pol = np.zeros((6),dtype='int')
    
    chain = sample.fix_pol(pol)
    
    import pdb; pdb.set_trace()
    
    IO = writer.stormpy_io(chain)
    IO.write()
    #IO.solve_PRISM()
    res = IO.solve()
    
    import pdb; pdb.set_trace()

if __name__=="__main__":
    main()
