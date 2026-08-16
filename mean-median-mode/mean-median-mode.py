import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    x = np.array(x)
    mean = np.mean(x)
    median = np.median(x)
    counts = Counter(x)
    max_freq = max(counts.values())
    possible_modes = [num for num, freq in counts.items() if freq == max_freq]
    mode = min(possible_modes)
    return float(mean), float(median), float(mode)