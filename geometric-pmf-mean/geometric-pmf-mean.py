import numpy as np

def geometric_pmf_mean(k, p):
    """
    Compute Geometric PMF and Mean.
    """
    # Write code here
    q = 1-p 
    mean = 1/p 
    arr = np.array(k)
    l = len(arr)
    ans = np.zeros(l)
    for i in range(l): 
      ans[i] = (q**(arr[i]-1))*p
    return ans,mean