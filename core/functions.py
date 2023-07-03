import numpy as np
import time
from scipy.stats import beta as betaF

def TicTocDifference():
    ''' Generator that returns time differences '''
    tf0 = time.time() # initial time
    tf = time.time() # final time
    while True:
        tf0 = tf
        tf = time.time()
        yield tf-tf0 # returns the time difference

TicTocDiff = TicTocDifference() # create an instance of the TicTocGen generator
    
def tocDiff(tempBool=True):
    ''' Print current time difference '''
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicTocDiff)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %np.round(tempTimeInterval, 5) )
    else:
        return np.round(tempTimeInterval, 12)
        
    return tempTimeInterval

def ticDiff():
    ''' Start time recorder '''
    # Records a time in TicToc, marks the beginning of a time interval
    tocDiff(False)

def computeBetaPPF(N, k, d, beta):
    
    epsilon = betaF.ppf(beta, k+d, N-(d+k)+1)
    
    return epsilon

def computeBetaCDF(N, k, d, epsilon):
    
    cum_prob = betaF.cdf(epsilon, k+d, N-(d+k)+1)
    
    return cum_prob

def compute_eta_satisfying(N, beta):
    
    return (1-beta)**(1/N)

def compute_beta_satisfying(N, eta_low):
    
    return 1 - eta_low ** N

def compute_eta_discard(N, nr_discarded, beta):
    '''
    Compute the lower bound on the satisfaction probability (eta) based on the
    confidence probability (beta)

    Parameters
    ----------
    N : int
        Number of samples.
    nr_discarded : int
        Number of discarded samples.
    beta : float
        Confidence probability.

    Returns
    -------
    eta_low : float
        Lower bound on the satisfaction probability.

    '''
    
    # Constraints discarded
    if nr_discarded >= N:
        eta_low = 0
        
    else:
        # Compute confidence level, compensated for the number of samples
        beta_bar = (1-beta)/N
        
        # Compute the lower bound on the satisfaction probability
        eta_low = 1 - computeBetaPPF(N, k=nr_discarded, d=1, 
                                     beta=1-beta_bar)
            
    return eta_low
            
def compute_beta_discard(N, nr_discarded, eta_low):
    '''
    Compute the confidence probability (beta) based on the lower bound on the
    satisfaction bound (eta)

    Parameters
    ----------
    N : int
        Number of samples.
    nr_discarded : int
        Number of discarded samples.
    eta_low : float
        Lower bound on the satisfaction probability.

    Returns
    -------
    beta : float
        Confidence probability.

    '''
    
    # Samples discarded
    if nr_discarded >= N:
        beta = 0
        
    else:

        # Compute the confidence level by which a given lower bound eta
        # is valid
        RHS = computeBetaCDF(N, k=nr_discarded, d=1, epsilon=1-eta_low)
        beta = 1 - N * (1-RHS)
        
    return max(0, beta)