import numpy.random as rnd

def gauss(mu, sigma):
    def gauss_sampler():
        return rnd.multivariate_normal(mu, sigma)
    return gauss_sampler
