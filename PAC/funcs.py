import numpy as np
from scipy.stats import beta as betaF

def calc_eta_var_thresh(beta, N):
    return (1-beta)**(1/N)

def calc_eta_discard(beta, N, discarded):
    if N == discarded:
        return 0
    else:    
        beta_bar = (1-beta)/N
        d = 1
        k = discarded
        return 1-betaF.ppf(1-beta_bar, k+d, N-(d+k)+1) 

def calc_eta_fixed_discard(beta, N, k):
    if N == k:
        return 0
    else:    
        beta_bar = (1-beta)
        d = 1
        return 1-betaF.ppf(1-beta_bar, k+d, N-(d+k)+1) 
    #return 1-k/N-(np.sqrt(k)/N+((np.sqrt(k)+1)/N)*np.log(1/(1-beta)))
