import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.array(y)
    n = len(y)
    if n == 0 : 
        return 0.0 
    val , cnt = np.unique(y, return_counts = True)
    probs = cnt/n 
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    return float(max(entropy,0.0))