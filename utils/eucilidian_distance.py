import numpy as np
import math

def eucilidian_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))