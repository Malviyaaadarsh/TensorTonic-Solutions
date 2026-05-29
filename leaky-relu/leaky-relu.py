import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    for i in range(len(x)):
        if x[i]<0:
            x[i] = alpha*x[i]
        else:
            continue
    return np.array(x) 