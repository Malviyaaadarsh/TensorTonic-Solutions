import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    m,n= len(A),len(A[0])  
    a_T = [[0]*m for _ in range(n)]   # n*m 
    for i in range(n):
        for j in range(m):
            a_T[i][j]= A[j][i]

    return np.array(a_T)
