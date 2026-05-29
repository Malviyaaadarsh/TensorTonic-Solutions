import numpy as np
def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    arr = np.array(x,dtype=float) 
    if arr.ndim ==1:
        exi = np.exp(arr-np.max(arr))
        return exi/np.sum(exi)
    elif arr.ndim==2: 
        exi = np.exp(arr-np.max(arr,axis=1,keepdims=True))
        return exi/np.sum(exi,axis=1,keepdims=True)
    