import numpy as np
from scipy.spatial import KDTree
from path_integral import compute_incremental_path_integral, compute_path_integral

# Euclidean distance (for now)
def dist(xi, xj):
    return np.linalg.norm(xi - xj)

def k_nearest_neighbors(data, k): # Do it once and return 3 and k
    tree = KDTree(data)  # Build KDTree
    distances, indices = tree.query(data, k=k+1)  # Query k+1 (includes self)
    return indices[:, 1:], distances[:, 1:]  # Remove self (0th neighbor)


def sigma_squared(data, knn3i, a): # a to be set
    n = data.shape[0]
    
    neighbors = data[knn3i]
    squared_dists = np.sum((data[:, None, :] - neighbors) ** 2, axis=2)  
    den = -3*n*np.log(a)

    return squared_dists / den


# i and j are indices
# knn is the indices of the K nearest neighbours. K to be set.
def pairwise_similarity(data, i, j, knn, sigma2):
    if j in knn[i]:
        np.exp(-dist(data[i],data[j]) ** 2 / sigma2)
    else:
        return 0
    
def get_edges(W):
    E = []
    n = W.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j:
                E.append(W[i,j])

    return np.array(E)

def compute_P(W):
    row_sums = np.sum(W, axis=1)  # Compute row sums (d_ii)
    
    # Avoid division by zero for isolated nodes
    np.place(row_sums, row_sums == 0, 1)

    # Compute D^(-1) * W using reciprocal for speed
    P = W * np.reciprocal(row_sums)[:, None]  # Element-wise multiplication

    return P

def create_digraph(X, k=3, a=0):
    X = np.asarray(X)
    n = X.shape[0]
    # Compute parameters
    knn_ind, knn_dis = k_nearest_neighbors(X,k)
    knn3_ind, knn3_dis = k_nearest_neighbors(X,3)
    sigma2 = sigma_squared(X,knn3_ind,a)
    
    #a = np.prod(W, axis=1) ** (1/3)

    # Weighted adjacency matrix
    W = np.fromfunction(np.vectorize(lambda i, j: pairwise_similarity(X,i,j,knn_ind,sigma2)), (n, n), dtype=int)

    return W

def run(X, nt):
    """ Runs Agglome1rative clustering via maximum incremental path integral.

    Args:
        X (_type_): set of n sample vectors X = {x1; x2;…; xn}
        n (_type_): number of clusters
    """
    
    W = create_digraph(X)
    P = compute_P(W)
    C = X # Initially, each point is its own cluster
    nc = C.shape[0]
        
    while nc > nt: # C_a, C_b
        S_Ca_given_CaUCb = compute_incremental_path_integral()
        A_Ca_Cb = 
        nc = nc - 1
        
    

if __name__ == "__main__":
    run()