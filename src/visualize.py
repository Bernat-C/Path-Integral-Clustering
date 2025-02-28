import numpy as np
import matplotlib.pyplot as plt

def visualize_clusters(X, clusters):
    """Visualizes clustered data in 2D."""
    plt.figure(figsize=(8, 6))
    
    for cluster_idx, points in enumerate(clusters):
        cluster_points = np.array([X[p] for p in points])  # Convert list to NumPy array
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_idx}")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Clustering Visualization")
    plt.legend()
    plt.show()