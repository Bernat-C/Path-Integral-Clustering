import numpy as np
import matplotlib.pyplot as plt

def visualize_clusters(X, clusters, active_clusters, title="Clustering Visualization"):
    """Visualizes clustered data in 2D."""
    #plt.ion()
    plt.figure(figsize=(8, 6))
    
    for cluster_idx, points in enumerate(clusters):
        if cluster_idx in active_clusters:
            cluster_points = np.array([X[p] for p in points])  # Convert list to NumPy array
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_idx}")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend()
    plt.show()