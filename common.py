from functools import cache

from scipy.stats import norm


@cache
def cache_cdf(z):
    return norm.cdf(z)


def fast_cdf(z):
    if z < -3:
        return 0
    if z > 3:
        return 1
    return cache_cdf(round(z, 10))
