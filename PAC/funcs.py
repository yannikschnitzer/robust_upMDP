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

def calc_eps_risk_complexity(beta, N, k):
    alphaL = betaF.ppf(beta, k, N-k+1)
    alphaU = 1-betaF.ppf(beta, N-k+1, k)

    m1 = np.expand_dims(np.arange(k, N+1),0)
    aux1 = np.sum(np.triu(np.log(np.ones([N-k+1,1])@m1),1),1)
    aux2 = np.sum(np.triu(np.log(np.ones([N-k+1,1])@(m1-k)),1),1)
    coeffs1 = np.expand_dims(aux2-aux1, 1)

    m2 = np.expand_dims(np.arange(N+1, 4*N+1),0)
    aux3 = np.sum(np.tril(np.log(np.ones([3*N,1])@m2)),1)
    aux4 = np.sum(np.tril(np.log(np.ones([3*N,1])@(m2-k))),1)
    coeffs2 = np.expand_dims(aux3-aux4, 1)

    def poly(t):
        return 1 + beta/(2*N) - (beta/(2*N))*np.sum(np.exp(coeffs1 - (N-m1.T)*np.log(t)))-(beta/(6*N))*np.sum(np.exp(coeffs2 + (m2.T-N)*np.log(t)))

    t1 = 1-alphaL
    t2 = 1
    poly1 = poly(t1)
    poly2 = poly(t2)


    if ((poly1*poly2)) > 0:
        epsL = 0
    else:
        while t2-t1 > 10**-10:
            t = (t1+t2)/2
            polyt  = poly(t)
            if polyt > 0:
                t1 = t
            else:
                t2 = t
        epsL = 1-t2

    t1 = 0
    t2 = 1-alphaU
    p1 = poly(t1)
    p2 = poly(t2)

    while t2-t1 > 10**-10:
        t = (t1+t2)/2
        polyt  = poly(t)
        if polyt > 0:
            t2 = t
        else:
            t1 = t
    epsU = 1-t1
   
    return epsL, epsU

