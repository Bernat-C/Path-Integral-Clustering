from matplotlib import pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from data import generate_synthetic, load_bc_wisconsin, load_mnist, load_usps, add_gaussian_noise, add_structural_noise
from kmeans import run_ap, runAC, zeta_function_clustering, diffusion_kernel_clustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import normalized_mutual_info_score
from pic import PathIntegralClustering
from metrics import clustering_error
from visualize import visualize_clusters

def test():
    n_samples = 500
    n_features = 10
    target_clusters = 10
    gaussian_noise_level = 1.8
    structural_noise_level = 0.3
    
    print("Generating data")
    X, y_true = generate_synthetic(n_samples=n_samples, n_features=n_features, centers=target_clusters ,random_state=42)
    X = add_gaussian_noise(X, gaussian_noise_level)
    X, y_true = add_structural_noise(X, y_true, structural_noise_level)
    #X, y_true = load_usps()
    #X, y_true = load_mnist()
    #X, y_true = load_bc_wisconsin()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Initializing clusters")
    y_pred = {}
    
    pic = PathIntegralClustering(target_clusters, z=0.01, a=0.95, K=20)
    y_pred["AP"] = run_ap(X_scaled)
    y_pred["A-Link"] = runAC(X_scaled, method='average')
    y_pred["S-link"] = runAC(X_scaled, method='single')
    y_pred["C-link"] = runAC(X_scaled, method='complete')
    y_pred["Zell"] = zeta_function_clustering(X_scaled, target_clusters)
    y_pred["D-kernel"] = diffusion_kernel_clustering(X_scaled, target_clusters)
    y_pred["PIC"] = pic.fit_predict(X_scaled)
    
    visualize_clusters(X_scaled, y_pred["PIC"], title="Definitive clusters")
    for method in ["PIC","AP","A-Link","S-link","C-link","Zell","D-kernel"]:
        print(f" #################### {method} ####################")
        # Compute and print Normalized Mutual Information Score (NMI)
        nmis = normalized_mutual_info_score(y_true, y_pred[method])
        print(f"Normalized mutual information score: {nmis:.4f}")
        
        # Compute and print Clustering Error (CE)
        ce = clustering_error(y_true, y_pred[method])
        print(f"Clustering error: {ce:.4f}")
        
        # Compute and print Silhouette Score
        silhouette = silhouette_score(X_scaled, y_pred[method]) 
        print(f"Silhouette score: {silhouette:.4f}")
        
        # Compute and print Davies-Bouldin Index
        davies_bouldin = davies_bouldin_score(X_scaled, y_pred[method])
        print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
        
        # Compute and print Calinski-Harabasz Index
        calinski_harabasz = calinski_harabasz_score(X_scaled, y_pred[method])
        print(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")


def create_datasets():
    dataset1, labels1 = generate_synthetic(n_samples=1000, centers=3, n_features=2, random_state=42)
    dataset2, labels2 = generate_synthetic(n_samples=1000, centers=4, n_features=2, random_state=42)
    dataset3, labels3 = generate_synthetic(n_samples=1000, centers=5, n_features=2, random_state=42)
    
    return [(dataset1, labels1), (dataset2, labels2), (dataset3, labels3)]

def evaluate_clustering(X, y_true, noise_level, noise_type="gaussian", n_clusters=3):
    nmi_scores = []
    
    for _ in range(20):  # Repeat for 20 times
        # Add noise to the data
        if noise_type == "gaussian":
            X_noisy = add_gaussian_noise(X, noise_level)
        elif noise_type == "structural":
            X_noisy = add_structural_noise(X, noise_level)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_noisy)
    
        # Apply clustering
        y_pred = {}
        pic = PathIntegralClustering(n_clusters, z=0.01, a=0.95, K=20)
        y_pred["AP"] = run_ap(X_scaled)
        y_pred["A-Link"] = runAC(X_scaled, method='average')
        y_pred["S-link"] = runAC(X_scaled, method='single')
        y_pred["C-link"] = runAC(X_scaled, method='complete')
        y_pred["Zell"] = zeta_function_clustering(X_scaled, n_clusters)
        y_pred["D-kernel"] = diffusion_kernel_clustering(X_scaled, n_clusters)
        y_pred["PIC"] = pic.fit_predict(X_scaled)
        
        # Compute NMI score
        nmi_score = {}
        for key, val in y_pred:
            nmi_score[key] = normalized_mutual_info_score(y_true, val)
        nmi_scores.append(nmi_score)
    
    # Return average NMI and standard deviation
    return np.mean(nmi_scores), np.std(nmi_scores)

def plot_results():
    datasets = create_datasets()  # Generate 3 datasets (I, II, III)
    noise_levels = [0.1, 0.2, 0.3, 0.4]  # Different noise levels for structural noise
    gaussian_std_devs = [1, 1.8]  # From snoise to 1.8 * snoise
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('NMI Scores with Gaussian and Structural Noise', fontsize=16)
    
    # Loop through datasets
    for i, (X, y_true) in enumerate(datasets):
        # Standardize dataset before clustering (important for distance-based methods)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Plot Gaussian noise results (a1, b1, c1)
        ax = axes[0, i]
        avg_nmi, std_nmi = evaluate_clustering(X_scaled, y_true, gaussian_std_devs[0], "gaussian")
        ax.bar(gaussian_std_devs[0], avg_nmi, yerr=std_nmi, capsize=5, label="Gaussian Noise")
        ax.set_title(f"Gaussian Noise - Dataset {i + 1}")
        ax.set_xlabel('Std Deviation of Noise')
        ax.set_ylabel('Average NMI')
        ax.set_ylim(0, 1)
        
        # Plot Structural noise results (a2, b2, c2)
        ax = axes[1, i]
        avg_nmi, std_nmi = evaluate_clustering(X_scaled, y_true, noise_level=0.2, noise_type="structural")
        ax.bar(0.2, avg_nmi, yerr=std_nmi, capsize=5, label="Structural Noise")
        ax.set_title(f"Structural Noise - Dataset {i + 1}")
        ax.set_xlabel('Proportion of Points Removed')
        ax.set_ylabel('Average NMI')
        ax.set_ylim(0, 1)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit title
    plt.show()

if __name__ == "__main__":
    test()