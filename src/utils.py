import numpy as np
from scipy.spatial import KDTree

# Euclidean distance (for now)
def dist(xi, xj):
    """ Computes the euclidean distance between two arrays

    Args:
        xi (np.array)
        xj (np.array)

    Returns:
        int: distance
    """
    return np.linalg.norm(xi - xj)

def k_nearest_neighbors(data, k): # Do it once and return 3 and k
    tree = KDTree(data)  # Build KDTree
    distances, indices = tree.query(data, k=k+1)  # Query k+1 (includes self)

    return indices[:, 1:], distances[:, 1:]  # Remove self (0th neighbor)