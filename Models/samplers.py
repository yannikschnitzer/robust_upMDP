import random

def gauss(mu, sigma):
    def gauss_sampler():
        return random.gauss(mu, sigma)
    return gauss_sampler
