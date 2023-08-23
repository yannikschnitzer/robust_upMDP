import numpy as np
from scipy.stats import beta as betaF
from tqdm import tqdm
import Markov.writer as writer
import Markov.models

def MC_sampler(model, k, thresh, pol=None):
    inn_count = 0
    for j in tqdm(range(k)):
        sample = Markov.models.MDP(model.sample_MDP())
        if pol is not None:
            sample = sample.fix_pol(pol)
        IO = writer.stormpy_io(sample)
        IO.write()
        res, all_res, sol_pol = IO.solve()
        if model.opt == "max":
            if res[0] < thresh:
                inn_count+= 1
        else:
            if res[0] > thresh:
                inn_count += 1
    
    violation_rate = inn_count/k
    
    return violation_rate

def MC_perturbed(model, k, thresh, pol=None, var=0.1):
    inn_count = 0
    for j in tqdm(range(k)):
        params = model.param_sampler()
        exact = model.fix_params(params)
        if pol is None:
            IO = writer.stormpy_io(exact)
            IO.write()
            res, all_res, pol = IO.solve()
        perturbed = params + np.random.normal(scale=var, size=np.size(exact))
        pert_model = model.fix_params(perturbed)
        pert_model = pert_model.fix_pol(pol)

        IO = writer.stormpy_io(pert_model)
        IO.write()
        res, all_res, sol_pol = IO.solve()
        if model.opt == "max":
            if res[0] < thresh:
                inn_count+= 1
        else:
            if res[0] > thresh:
                inn_count += 1
    
    violation_rate = inn_count/k
    
    return violation_rate

def calc_eps(beta, N, d):
    eps = betaF.ppf(1-beta, d, N-d+1)
    return eps

def calc_eta_var_thresh(beta, N):
    return calc_eps(beta, N, 1)
    #return 1-(beta)**(1/N)

def calc_eta_discard(beta, N, discarded):
    if N == discarded:
        return 0
    else:    
        beta_bar = (beta)/N
        d = 1
        k = discarded
        return betaF.ppf(1-beta_bar, k+d, N-(d+k)+1) 

def calc_eta_fixed_discard(beta, N, k):
    if N == k:
        return 0
    else:    
        beta_bar = (beta)
        d = 1
        return betaF.ppf(1-beta_bar, k+d, N-(d+k)+1) 
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

