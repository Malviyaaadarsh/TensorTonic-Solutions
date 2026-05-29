import numpy as np
import math
from scipy.special import erf
def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    X = np.array(x,dtype=float)
    for i in range(len(X)):
        X[i] = 0.5 * X[i] * (1 + erf(X[i]/np.sqrt(2)))
    return X 