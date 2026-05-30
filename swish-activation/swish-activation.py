import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    X = np.array(x,dtype=float)
    for i in range(len(X)):
        X[i] = X[i] / (1+np.exp(-X[i]))
    return X