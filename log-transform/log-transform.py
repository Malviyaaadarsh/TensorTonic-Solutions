import math 
def log_transform(values):
    """
    Apply the log1p transformation to each value.
    """
    for i in range(len(values)):
        values[i] = math.log1p(values[i])
    return values