import numpy as np
from core.functions import compute_eta_satisfying, compute_beta_satisfying, \
    compute_eta_discard, compute_beta_discard

def compute_avg_beta(violated_array, N, eta, threshold,
                     verbose=False, precision=5):
    '''
    Compute the average confidence level (beta), given the violations for a 
    number of iterations, and given a fixed satisfaction probability (eta)
    
    Parameters
    ----------
    counterarray : Approximate satisfaction probs
    N : Number of samples
    eta : Lower bound on the satisfaction probability
    verbose : If true, additional outputs are shown
    precision : Number of decimals to round to
    
    Returns
    -------
    beta_mean : Average confidence level (beta)
    
    '''
    
    if type(threshold) == bool:
    
        beta = compute_beta_satisfying(N, eta)
        
        print("Low confidence probability (beta) for eta="+
              str(eta)+" is: ",str(beta)+' (obtained from Theorem 1)')
        
    else:   
    
        #storing probabilities for each iteration
        beta_array = np.zeros(len(violated_array))
    
        for i in range(len(violated_array)):
    
            #compute number of constraints to remove in the LP
            removeconstraints = violated_array[i]
            
            #approximately compute the confidence prob
            beta = compute_beta_discard(N, removeconstraints, eta)
    
            if verbose:
                print('Beta for N',N,' k',removeconstraints,' eta',eta,' is',beta)
    
            beta_array[i] = beta
            
        beta = np.mean(beta_array)
            
        print("Avg. confidence probability (beta) for eta="+str(eta)+" is:",
              beta)
    
    return np.round(beta, precision)
    
def compute_avg_eta(violated_array, N, beta, threshold,
                    verbose=False, precision=5):
    '''
    Compute the average satisfaction probability (eta), given the violations 
    for a number of iterations, and given a fixed confidence level (beta)
    
    Parameters
    ----------
    counterarray : Approximate satisfaction probs
    N : Number of samples
    beta : Confidence level
    verbose : If true, additional outputs are shown
    precision : Number of decimals to round to
    
    Returns
    -------
    eta_mean : Average satisfaction probability (eta)
    
    '''
    
    if type(threshold) == bool:
    
        eta = compute_eta_satisfying(N, beta)
        
        print("Low bound on the satisfaction probability (eta) for beta="+
              str(beta)+" is: "+str(eta)+' (obtained from Theorem 1)')
        
    else:    
    
        #storing probabilities for each iteration
        eta_array = np.zeros(len(violated_array))
    
        for i in range(len(violated_array)):
    
            #compute number of constraints to remove in the LP
            removeconstraints = violated_array[i]
            
            #approximately compute the confidence prob
            eta = compute_eta_discard(N, removeconstraints, beta)
    
            if verbose:
                print('Eta for N',N,' k',removeconstraints,' beta',beta,' is',eta)
    
            eta_array[i] = eta
        
        eta = np.mean(eta_array)
        
        print("Avg. low bound on the satisfaction probability (eta) for beta="+
              str(beta)+" is:",eta)

    return np.round(eta, precision)