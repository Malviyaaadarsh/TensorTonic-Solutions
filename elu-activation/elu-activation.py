import numpy as np
def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    for i in range(len(x)):
        x[i] = x[i] if x[i]>0 else alpha*(np.exp(x[i])-1)
    return x