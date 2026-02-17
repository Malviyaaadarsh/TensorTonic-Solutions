import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    mean_x = np.mean(x)
    n = len(x)
    var = np.sum((x-mean_x)**2)/(n-1)
    sd = np.sqrt(var)
    return var,sd