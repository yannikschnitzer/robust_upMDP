import scipy.stats as stats
import numpy as np

def gauss(mu, sigma, lower=0.01, upper=0.99):
    def gauss_sampler():
        out = np.zeros_like(mu)
        for i, elem in enumerate(mu):
            if type(lower) == float:
                sample = stats.truncnorm.rvs(lower, upper, sigma[i,i], elem)
            else:
                sample = stats.truncnorm.rvs(lower[i], upper[i], sigma[i,i], elem)
            out[i] = sample
        return out
    return gauss_sampler

def uniform(size, lower=0.01, upper=0.99):
    def uniform_sampler():
        if type(lower) == float:
            return np.random.uniform(lower, upper, size=size)
        else:
            return np.array([np.random.uniform(lower[i], upper[i]) for i in range(size)])

    return uniform_sampler
