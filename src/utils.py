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

def k_nearest_neighbors_and_3(data, k):
    """Computes k nearest neighbours and 3 nearest neighbours.

    Args:
        data (np.array): Input data.
        k (int): Number of neighbours to query.

    Returns:
        dict: {
            'k_neighbors': (indices, distances),
            '3_neighbors': (indices, distances)
        }
    """
    max_k = max(k, 3)
    tree = KDTree(data)
    distances, indices = tree.query(data, k=max_k + 1)  # Query k+1 (includes self)

    # Remove self (0th neighbor)
    indices, distances = indices[:, 1:], distances[:, 1:]

    return indices[:, :k] , distances[:, :k], indices[:, :3], distances[:, :3]
