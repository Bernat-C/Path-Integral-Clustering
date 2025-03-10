import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def visualize_clusters(X, y, title="Clustering Visualization"):
    """Visualizes clustering results in 2D

    Args:
        X (np.array)
        y (np.array)
        title (str, optional). Defaults to "Clustering Visualization".
    """
    
    #plt.ion()
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Create a scatter plot of the clusters
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, marker='o', edgecolor='k')

    # Add labels and title
    plt.title(title, fontsize=14)
    plt.xlabel("PCA Component 1", fontsize=12)
    plt.ylabel("PCA Component 2", fontsize=12)

    # Add a color bar to indicate cluster labels
    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster", fontsize=12)

    # Show the plot
    plt.show()