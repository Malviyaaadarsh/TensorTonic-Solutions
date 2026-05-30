import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    X = np.array(x,dtype=float)
    for i in range(len(X)):
        X[i] = (np.exp(X[i])-np.exp(-X[i]))/(np.exp(X[i])+np.exp(-X[i]))
    return X
#   return np.tanh(x)