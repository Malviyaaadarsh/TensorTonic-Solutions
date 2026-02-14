import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    q = 1-p
    pmf = (comb(n,k)) * (p**k) * (q**(n-k))
    cdf = 0
    for i in range(0,k+1):
      cdf += (comb(n,i))*(p**i)*(q**(n-i))
    return pmf,cdf  