import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    x = np.array(x)
    m = np.mean(x)
    mid = np.median(x)
    cnt = Counter(x)
    maxi = max(cnt.values())
    mod = [key for key , freq in cnt.items() if freq==maxi]
    return (float(m),float(mid),float(min(mod)))