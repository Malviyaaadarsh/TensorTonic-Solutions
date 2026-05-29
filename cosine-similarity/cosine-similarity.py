import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    my_a = np.linalg.norm(a)
    my_b = np.linalg.norm(b)
    if my_a == 0 or my_b == 0: 
        return 0.0
    else : 
        return float((np.dot(a,b)/(my_a*my_b)))