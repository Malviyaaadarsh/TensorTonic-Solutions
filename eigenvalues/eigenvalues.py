import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    try : 
        
        m = np.asarray(matrix,dtype=float) 
        if m.shape[0]!=m.shape[1]:
            return None
        else :
            return (np.linalg.eigvals(m))
    except : 
        return None 