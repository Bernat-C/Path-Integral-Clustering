import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

PLOT_BLOCK_EXECUTION = True

def visualize_clusters(X, y, title="Clustering Visualization", save_path = False):
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

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show(block=PLOT_BLOCK_EXECUTION)
        
def plot_noise_results(results: dict, x_indices, noise_type="Gaussian", save_path = False):
    """ Makes a line plot of the noise results

    Args:
        results (dict): Dictionary containing all the algorithms with each mean and standard deviation
        x_indices (_type_): Noise levels
        save_path (bool, optional): Path to save the image. Defaults to False.
    """
    methods = list(results[0].keys())

    plt.figure(figsize=(12, 6))

    for method in methods:
        means = [d[method][0] for d in results]
        stds = [d[method][1] for d in results]
        
        plt.errorbar(
            x_indices,
            means,
            yerr=stds,
            label=method,
            capsize=4,
            marker='o',
            linestyle='-'
        )

    # Customize
    plt.xlabel(f'{noise_type} Noise Level')
    plt.ylabel('Score')
    plt.title(f'Performance vs. {noise_type} Noise Level')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show(block=PLOT_BLOCK_EXECUTION)
        plt.close()