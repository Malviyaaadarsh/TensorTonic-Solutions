import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    pred = np.array(y_pred)
    true = np.array(y_true)
    N = len(pred)   
    sum = 0.0
    for i in range(N):
        sum += (pred[i]-true[i])**2 
    mean = sum/N 
    return mean 
        