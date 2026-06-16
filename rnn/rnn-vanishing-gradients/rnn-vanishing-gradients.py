import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """
    spectral_norm = np.linalg.norm(W_hh,ord=2)
    gradient_norm = [1.0]
    for _ in range(1,T):
        gradient_norm.append(spectral_norm*gradient_norm[-1])
    return gradient_norm 