from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, KMeans
from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.metrics import pairwise_distances
from scipy.sparse.linalg import eigsh
import numpy as np

# 1. Affinity Propagation (AP)
def run_ap(X):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    ap = AffinityPropagation(random_state=42)
    ap_labels = ap.fit_predict(X)
    
    return ap_labels

def runAC(X, target_clusters, method='average'):
    # Runs Agglomerative Clustering with Average Linkage (A-Link) and S-Link and C-Link
    agglo_avg = AgglomerativeClustering(n_clusters=target_clusters, metric='euclidean', linkage=method) # average, single, complete
    agglo_avg_labels = agglo_avg.fit_predict(X)
    return agglo_avg_labels
    
def diffusion_kernel_clustering(X, n_clusters, alpha=0.95, z=0.01):
    """Diffusion Kernel-based Clustering using von Neumann kernel."""
    kernel = von_neumann_kernel(X, alpha=alpha, z=z)
    
    # Apply average linkage clustering
    model = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    return model.fit_predict(1 - kernel)  # Convert similarity to distance

def von_neumann_kernel(X, alpha=0.95, z=0.01):
    """Compute the von Neumann kernel."""
    n = X.shape[0]
    dists = pairwise_distances(X, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    W = 1.0 / np.power(dists, z)
    W[~np.isfinite(W)] = 0.0  # Replace inf/nan with 0
    
    # Create diagonal matrix D
    D = np.diag(np.sum(W, axis=1))
    
    # Normalized Laplacian
    L = np.eye(n) - np.linalg.inv(np.sqrt(D)) @ W @ np.linalg.inv(np.sqrt(D))
    
    # Compute the kernel
    K = np.linalg.inv(np.eye(n) + z * L)
    
    # Apply alpha parameter for diffusion
    K_alpha = np.linalg.matrix_power(K, int(1/(1-alpha)))
    
    return K_alpha

def zeta_function_clustering(X, n_clusters, z=0.01):
    """Zeta Function-based Clustering."""
    n_samples = X.shape[0]

    # Step 1: Compute pairwise distances efficiently
    dists = pairwise_distances(X, metric='euclidean')

    # Step 2: Compute similarity matrix using Zeta kernel
    np.fill_diagonal(dists, np.inf)  # avoid self-distance = 0
    sim = 1.0 / np.power(dists, z)
    
    # Replace any infinite or NaN values with 0
    sim[~np.isfinite(sim)] = 0.0

    # Step 3: Construct normalized graph Laplacian
    degrees = sim.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-8))  # avoid divide by zero
    L = np.eye(n_samples) - D_inv_sqrt @ sim @ D_inv_sqrt

    # Step 4: Compute the first k eigenvectors of L (skip the first trivial eigenvector)
    _, eigvecs = eigsh(L, k=n_clusters + 1, which='SM')  # smallest magnitude
    embedding = eigvecs[:, 1:n_clusters+1]

    # Step 5: KMeans clustering in spectral space
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(embedding)