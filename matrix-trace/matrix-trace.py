import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    ans = 0
    for i in range(len(A)): 
        for j in range(len(A[i])):
            if i == j : 
                ans = ans + A[i][j]
    return ans 
