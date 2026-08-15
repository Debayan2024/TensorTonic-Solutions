import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    # Convert input to a NumPy array
    x = np.array(x)
    
    # 1. Compute mean and median using NumPy
    mean = np.mean(x)
    median = np.median(x)
    
    # 2. Compute mode using Counter
    counts = Counter(x)
    
    # Fix: Call values() with empty parentheses to get the highest frequency
    max_freq = max(counts.values())
    
    # Fix: Find all values that match the highest frequency, then pick the smallest one
    possible_modes = [num for num, freq in counts.items() if freq == max_freq]
    mode = min(possible_modes)
    
    # 3. Return all three metrics as floats
    return float(mean), float(median), float(mode)