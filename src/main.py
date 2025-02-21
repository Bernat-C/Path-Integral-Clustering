import numpy as np
from scipy.spatial import KDTree

# Euclidean distance (for now)
def dist(xi, xj):
    return np.linalg.norm(xi - xj)

def k_nearest_neighbors(data, k):
    tree = KDTree(data)  # Build KDTree
    distances, indices = tree.query(data, k=k+1)  # Query k+1 (includes self)
    return indices[:, 1:], distances[:, 1:]  # Remove self (0th neighbor)


def sigma_squared(data, knn3i, a=0): # a to be set
    n = data.shape[0]
    for i in range(n):
        for j in knn3i:
            dist = dist(data[i],data[j]) ** 2
    den = -3*n*np.log(a)


# i and j are indices
# knn is the indices of the K nearest neighbours. K to be set.
def pairwise_similarity(data, i, j, knn, sigma2):
    if j in knn[i]:
        np.exp(-dist(data[i],data[j]) ** 2 / sigma2)
    else:
        return 0

def create_digraph(X):
    #W = # Weighted adjacency matrix
    n = W.shape[0]
    knn3_ind, knn3_dis = k_nearest_neighbors(X,3)
    weights = np.array([[W[i,j] for j in knn3_ind[i]] for i in range(n)])
    
    # Compute geometric mean: (w1 * w2 * w3)^(1/3)
    a = np.prod(weights, axis=1) ** (1/3)

    #sigma2 = sigma_squared(data, knn3i)
    #a = np.prod(w) ** (1 / len(w))  # Geometric mean
    pass