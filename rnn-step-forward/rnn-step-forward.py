import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """ 
    h_t = ( x_t @ Wx ) + ( h_prev @ Wh ) + b 
    h_t = np.tanh(h_t)
    return h_t 
