import tqdm
import numpy as np
from scipy.spatial.distance import cdist

def find_nearest_neighbors(X):
    """ find nearest neighbours using euclidean distance

    Args:
        X (np.array)

    Returns:
        _type_: _description_
    """
    distances = cdist(X, X)
    np.fill_diagonal(distances, np.inf)  # Ignore self-distances
    nearest_indices = np.argmin(distances, axis=1)  # Get nearest neighbors' indices
    return nearest_indices

def initialize_clusters(X):
    """Initializes the clusters using nearest neighbour initialization

    Args:
        X (_type_)

    Returns:
        C
    """
    nearest_indices = find_nearest_neighbors(X)
    n_samples = len(X)
    
    # Initialize clusters as a numpy array of shape (n_samples, 2)
    clusters = np.array([[i, nearest_indices[i]] for i in range(n_samples)])
    return clusters

def merge_clusters(clusters):
    """ Runs nearest neighbour initialization

    Args:
        clusters (np.array)

    Returns:
        final clusters
    """
    merged = True
    while merged:
        merged = False
        # Create an array to track which clusters are already merged
        to_merge = np.zeros(len(clusters), dtype=bool)
        
        new_clusters = []
        for i in range(len(clusters)):
            if to_merge[i]:  # If this cluster has already been merged, skip it
                continue
            
            cluster = clusters[i]
            merged_cluster = None
            for j in range(i + 1, len(clusters)):
                if to_merge[j]:  # If the other cluster is already merged, skip it
                    continue
                # Check for intersection (any common sample between clusters)
                if np.intersect1d(cluster, clusters[j]).size > 0:
                    # Merge the clusters by combining them
                    merged_cluster = np.union1d(cluster, clusters[j])
                    to_merge[j] = True
                    break
            
            if merged_cluster is not None:
                # Add merged cluster to the new clusters list
                new_clusters.append(merged_cluster)
                merged = True
            else:
                # If no merge, retain the current cluster
                new_clusters.append(cluster)
        
        # Update clusters with the merged ones
        clusters = new_clusters
    
    return clusters

def cluster_init(X):
    """ Generates initial clusters using nearest neighbours merging.

    We use a simple nearest neighbor merging algorithm to obtain initial clusters. 
    First, each sample and its nearest neighbor form a cluster and we obtain 
    n clusters, each of which has two samples.
    Then, the clusters are merged to remove duplicated samples,
    i.e., we merge two clusters if their intersection is nonempty, until the
    number of clusters cannot be reduced.

    """
    # Step 1: Initialize clusters with their neighbour
    clusters = initialize_clusters(X)
    
    # Step 2: Merge clusters until no more merges can be done
    final_clusters = merge_clusters(clusters)
    
    return final_clusters