import numpy as np
import heapq
from scipy.spatial import KDTree
from path_integral import compute_incremental_path_integral, compute_path_integral
from sklearn.metrics import normalized_mutual_info_score
from data import make_blobs
from visualize import visualize_clusters

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

def compute_affinity(Ca, Cb): # Equation 3
    """ Affinity

    Args:
        Ca (np.array): [[x1],[x2]...[xn]]
        Cb (np.array): [[x1],[x2]...[xn]]

    Returns:
        float: affinity
    """
    S_Ca = compute_path_integral()
    S_Cb = compute_path_integral()
    S_Ca_given_CaUCb = compute_incremental_path_integral()
    S_Cb_given_CaUCb = compute_incremental_path_integral()

    return (S_Ca_given_CaUCb - S_Ca) + (S_Cb_given_CaUCb - S_Cb)

def union(Ca, Cb):
    """ Performs Ca U Cb

    Args:
        Ca (np.array)
        Cb (np.array)

    Returns:
        np.array
    """
    return np.concatenate((Ca,Cb))

def run(X, nt):
    """ Runs Agglomerative clustering via maximum incremental path integral.

    Args:
        X (np.array): set of n sample vectors X = {x1; x2;…; xn}
        nt (int): target number of clusters
    """
    
    W = create_digraph(X)
    P = compute_P(W)

    # C should be computed using nearest neighbour merging.
    C = X.reshape(-1, 1) # Initially, each point is its own cluster [[x1],[x2],...[xn]]
    nc = C.shape[0]
        
    # Find clusters inside C that maximize the affinity measure
    if nc < 2:
        return C

    max_heap = []
    for i in range(nc):
        for j in range(i + 1, nc):
            affinity = compute_affinity(C[i], C[j])
            heapq.heappush(max_heap, (-affinity, i, j))  # Store negative affinity for max heap
    
    while nc > nt and len(C) > 1:
        # Get the most similar pair
        _, i, j = heapq.heappop(max_heap)
        if i >= len(C) or j >= len(C):  # Ignore if index is outdated due to previous merges
            continue

        # Merge the two elements
        merged = union(C[i], C[j])

        # Remove old elements and add merged element
        del C[max(i, j)]  # Remove the larger index first
        del C[min(i, j)]
        C.append(merged)

        # Update affinities with the new element
        new_idx = nc - 1
        for k in range(new_idx):  # Compute affinity with all previous elements
            affinity = compute_affinity(C[k], merged)
            heapq.heappush(max_heap, (-affinity, k, new_idx))  # Push new affinities to the heap
        nc = nc - 1
    
    return C
    

if __name__ == "__main__":
    ninstances = 300
    nt = 3

    data = make_blobs(ninstances,nt)

    C = run(data, nt)

    visualize_clusters(data, C)
