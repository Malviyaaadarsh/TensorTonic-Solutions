import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    ex = 0 
    n = len(x)
    psum = 0 
    for i in range(0,n):
      psum += p[i]
      ex = ex + x[i]*p[i]
    if not np.isclose(psum, 1.0, atol=1e-6):  # tolerane = 10^-6
        raise ValueError("Probabilities do not sum to 1.")
    return ex 