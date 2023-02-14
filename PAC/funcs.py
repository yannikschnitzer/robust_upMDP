import numpy as np
from scipy.stats import beta as betaF
from tqdm import tqdm
import Markov.writer as writer

def MC_sampler(model, k, N, thresh, violation_prob, pol=None):
    out_count = 0
    inn = []
    for i in tqdm(range(N)):
        inn_count = 0
        for j in range(k):
            sample = model.sample_MDP()
            if pol is not None:
                sample = sample.fix_pol(pol)
            IO = writer.stormpy_io(sample)
            IO.write()
            res, all_res = IO.solve()
            if res[0] < thresh:
                inn_count+= 1
        inn_prob = inn_count/k
        inn += [inn_prob]
        if inn_prob > violation_prob:
            out_count += 1
    out_prob = out_count/N
    avg_inn = sum(inn)/N
    max_inn = max(inn)
    min_inn = min(inn)
    return 1-out_prob, min_inn, avg_inn, max_inn

def calc_eta_var_thresh(beta, N):
    return 1-(1-beta)**(1/N)

def calc_eta_discard(beta, N, discarded):
    if N == discarded:
        return 0
    else:    
        beta_bar = (1-beta)/N
        d = 1
        k = discarded
        return betaF.ppf(1-beta_bar, k+d, N-(d+k)+1) 

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
    m1[0,0] = k+1
    aux1 = np.sum(np.triu(np.log(np.ones([N-k+1,1])@m1),1),1)
    aux2 = np.sum(np.triu(np.log(np.ones([N-k+1,1])@(m1-k)),1),1)
    coeffs1 = np.expand_dims(aux2-aux1, 1)

    m2 = np.expand_dims(np.arange(N+1, 4*N+1),0)
    aux3 = np.sum(np.tril(np.log(np.ones([3*N,1])@m2)),1)
    aux4 = np.sum(np.tril(np.log(np.ones([3*N,1])@(m2-k))),1)
    coeffs2 = np.expand_dims(aux3-aux4, 1)

    def poly(t):
        val = 1
        val += beta/(2*N) 
        val -= (beta/(2*N))*np.sum(np.exp(coeffs1 - (N-m1.T)*np.log(t)))
        val -=(beta/(6*N))*np.sum(np.exp(coeffs2 + (m2.T-N)*np.log(t)))

        
        return val
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

    while t2-t1 > 10**-10:
        t = (t1+t2)/2
        polyt  = poly(t)
        if polyt > 0:
            t2 = t
        else:
            t1 = t
    epsU = 1-t1
   
    return epsL, epsU

