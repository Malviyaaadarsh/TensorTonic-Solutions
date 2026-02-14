import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    # Write code here
    q = 1-p 
    arr= np.array(x)
    l = len(arr)
    ans = np.zeros(l)
    mean = p
    var= (p*q)
    for i in range(l):
      if arr[i]==1: 
        ans[i]= p 
      else : 
        ans[i]= q 
    return ans,mean,var