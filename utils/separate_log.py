import numpy as np

def separate_log(x):
    if x > 0:
        return np.log(x)
    elif x < 0:
        return -np.log(abs(x))
    elif x == 0:
        return 0  # Handle zero separately if needed
    else:
        return np.nan