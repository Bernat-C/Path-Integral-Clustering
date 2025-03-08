import numpy as np
import heapq
import tqdm
from scipy.spatial import KDTree
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import confusion_matrix

from path_integral import compute_incremental_path_integral, compute_path_integral, compute_incremental_path_integral_inefficient
from visualize import visualize_clusters
from data import generate_synthetic, load_usps
from nearest_neighbour_init import cluster_init

# Euclidean distance (for now)
def dist(xi, xj):
    return np.linalg.norm(xi - xj)

def k_nearest_neighbors(data, k): # Do it once and return 3 and k
    tree = KDTree(data)  # Build KDTree
    distances, indices = tree.query(data, k=k+1)  # Query k+1 (includes self)

    return indices[:, 1:], distances[:, 1:]  # Remove self (0th neighbor)


def sigma_squared(data, knn3_dis, a):
    n = data.shape[0]
    
    squared_dists = np.sum(np.square(knn3_dis))
    den = 3*n*(-np.log(a))
        
    return squared_dists / den

# knn is the indices of the K nearest neighbours. K to be set.    
def pairwise_similarity_matrix(data, knn, sigma2):
    n = len(data)
    W = np.zeros((n, n))  # Initialize the similarity matrix

    for i in tqdm.tqdm(range(n),desc="Computing matrix W"):
        for j in knn[i]:  # Only compute for k-nearest neighbors
            W[i, j] = np.exp(-(dist(data[i], data[j]) ** 2 / sigma2))

    return W
    
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

def create_digraph(X, k, a):
    print(f"Creating Digraph")
    X = np.asarray(X)
    n = X.shape[0]
    # Compute parameters
    knn_ind, knn_dis = k_nearest_neighbors(X,k)
    knn3_ind, knn3_dis = k_nearest_neighbors(X,3)
    sigma2 = sigma_squared(X,knn3_dis,a)
        
    # Weighted adjacency matrix
    W = pairwise_similarity_matrix(X, knn_ind, sigma2)

    return W

def compute_affinity_single_link(D,Ca,Cb):
    return -np.min([D[i, j] for i in Ca for j in Cb])

def compute_affinity(P, Ca, Cb, z): # Equation 3
    """ Affinity

    Args:
        Ca (np.array): [[1],[2]...[n]]
        Cb (np.array): [[1],[2]...[n]]

    Returns:
        float: affinity
    """
    Pa = P[np.ix_(Ca,Ca)]
    Pb = P[np.ix_(Cb,Cb)]

    S_Ca = compute_path_integral(Pa,z)
    #print(f"S_Ca: {S_Ca}")
    S_Cb = compute_path_integral(Pb,z)
    #print(f"S_Cb: {S_Cb}")
    S_Ca_given_CaUCb = compute_incremental_path_integral(Ca,Cb,P,z)
    #print(f"SCa given CaUb: {S_Ca_given_CaUCb}")
    S_Cb_given_CaUCb = compute_incremental_path_integral(Cb,Ca,P,z)
    #print(f"SCb given CaUb: {S_Cb_given_CaUCb}")

    result = (S_Ca_given_CaUCb - S_Ca) + (S_Cb_given_CaUCb - S_Cb)
    return result

def transform_to_assignments(clusters):
    # Create an empty list to hold the cluster assignments
    assignments = []
    
    # Iterate over each cluster and assign the cluster number to each element
    for cluster_id, cluster in enumerate(clusters):
        for index in cluster:
            # Add the cluster_id (representing the cluster number) to the assignments list
            assignments.append((index, cluster_id))
    
    # Sort assignments based on the original indices
    assignments.sort(key=lambda x: x[0])
    
    # Extract the cluster ids for the sorted list of indices
    result = [cluster_id for index, cluster_id in assignments]
    
    return result

def clustering_error(y_true, y_pred):
    """
    Computes the clustering error
    
    Parameters:
        y_true (numpy.ndarray): Ground truth labels.
        y_pred (numpy.ndarray): Cluster labels from the algorithm.
    
    Returns:
        float: Clustering error (misclassification rate).
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Solve the assignment problem (Hungarian algorithm) to maximize correct assignments
    row_ind, col_ind = linear_sum_assignment(-cm)  # Maximize correct pairs

    # Compute the total number of correctly assigned points
    correct = cm[row_ind, col_ind].sum()
    
    # Compute clustering error
    error = 1 - (correct / len(y_true))
    return error

def run(X, C, nt, z=0.01, a=0.95, K=20):
    """ Runs Agglomerative clustering via maximum incremental path integral.

    Args:
        X (np.array): set of n sample vectors X = {x1; x2;…; xn}
        C (np.array): cluster initializations
        nt (int): target number of clusters
    """
    assert(K<len(X))
    
    W = create_digraph(X,k=K,a=a) # a can't be 0 nor 1 (inf)
    P = compute_P(W)

    #D = cdist(X, X, metric="euclidean") # Used for trials using single-link instead of path integral
    
    nc = len(C)
    if nc < 2:
        return C # Nothing to merge
    
    max_heap = []
    for i in tqdm.tqdm(range(nc),desc="Computing initial cluster's affinities."):
        for j in range(i + 1, nc):
            affinity = compute_affinity(P, C[i], C[j], z)#compute_affinity_single_link(D,C[i],C[j])
            heapq.heappush(max_heap, (-affinity, i, j))  # Store negative affinity for max heap
            
    active_clusters = set(range(nc))
    
    print("The clustering process has begun")
    for i in tqdm.tqdm(range(len(active_clusters), nt, -1), desc="Clustering..."):
        
        # Get the most similar pair
        while max_heap:
            _, i, j = heapq.heappop(max_heap)
            #print(f"{i} - {j} is a candidate.")
            if i in active_clusters and j in active_clusters:
                # print(f"Cluster {i} {j} are the most affine.")
                break  # Found valid cluster pair
        else:
            break
        
        #visualize_clusters(X,C,active_clusters)
        merged = np.append(C[i], C[j])

        # Remove old elements and add merged element
        active_clusters.remove(i)
        active_clusters.remove(j)
        C.append(merged)
        new_idx = len(C) - 1
        active_clusters.add(new_idx)
        
        #print(f"Merged clusters {i} and {j} -> New cluster {new_idx}")

        # Update affinities with the new element
        for k in active_clusters - {new_idx}:  # Compute affinity with all previous elements excluding itself
            affinity =  compute_affinity(P, C[k], merged, z)#compute_affinity_single_link(D,C[k],merged)
            heapq.heappush(max_heap, (-affinity, k, new_idx))  # Push new affinities to the heap
    
    #print(f"Ended run with |C|:{len(C)} |AC|:{len(active_clusters)} with nt:{nt}")
    return [C[i] for i in active_clusters]

def test_synthetic():
    n_samples = 500
    n_features = 2
    nt = 10
    
    print("Generating data")
    data, y_true = generate_synthetic(n_samples=n_samples, n_features=n_features, centers=nt)

    print("Initializing clusters")
    C = cluster_init(data)
    #visualize_clusters(data, C, range(len(C)), title="Clustering initialization")
    
    C = run(data,C,nt,z=0.01,a=0.95,K=20)

    visualize_clusters(data, C, range(len(C)), title="Definitive clusters")

    y_pred = transform_to_assignments(C)
    nmis = normalized_mutual_info_score(y_true,y_pred)
    ce = clustering_error(y_true,y_pred)
    print(f"Normalized mutual information score {nmis}")
    print(f"Clustering error {ce}")
    
def test_usps():
    nt = 10
    
    print("Generating data")
    data, y_true = load_usps()

    print("Initializing clusters")
    C = cluster_init(data)
    #visualize_clusters(data, C, range(len(C)), title="Clustering initialization")
    
    C = run(data,C,nt,z=0.01,a=0.95,K=20)

    visualize_clusters(data, C, range(len(C)), title="Definitive clusters")

    y_pred = transform_to_assignments(C)
    nmis = normalized_mutual_info_score(y_true,y_pred)
    ce = clustering_error(y_true,y_pred)
    print(f"Normalized mutual information score {nmis}")
    print(f"Clustering error {ce}")
    
if __name__ == "__main__":
    test_synthetic()