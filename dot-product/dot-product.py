import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    # Write code here
    dot = np.dot(np.array(x), np.array(y))
    return float(dot)
    pass