import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    batch,T,input_dim = X.shape 
    h_t = h_0 
    hidden_dim = h_0.shape[1]
    hidden_states = np.zeros((batch, T, hidden_dim)) 
    for t in range(T):
        X_t = X[:,t,:] 
        h_t = np.tanh(X_t @ W_xh.T + h_t @ W_hh.T + b_h)
        hidden_states[:,t,:]= h_t 
    h_final = h_t 
    return (hidden_states,h_final)