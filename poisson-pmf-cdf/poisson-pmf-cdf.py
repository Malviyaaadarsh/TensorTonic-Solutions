import numpy as np

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    pmf = np.exp(-lam)
    cdf= pmf
    for i in range(1,k+1):
      pmf *= lam/i 
      cdf+=pmf
    return pmf,cdf   