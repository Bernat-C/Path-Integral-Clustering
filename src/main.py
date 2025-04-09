import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import tqdm
from config import Config
from data import generate_synthetic, load_bc_wisconsin, load_mnist, load_usps, add_gaussian_noise, add_structural_noise
from algorithms import run_ap, runAC, zeta_function_clustering, diffusion_kernel_clustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import normalized_mutual_info_score
from pic import PathIntegralClustering
from metrics import clustering_error
from visualize import visualize_clusters
import pickle as pkl
from pathlib import Path

DATA_DIR =  Path(__file__).parent / '..' / 'data'

def get_dataset(dataset_name, params=None):
    if dataset_name == "usps":
        return load_usps()
    elif dataset_name == "mnist":
        return load_mnist()
    elif dataset_name == "bc_wisconsin":
        return load_bc_wisconsin()
    elif dataset_name == "synthetic":
        generate_synthetic(n_samples=params['n_samples'], n_features=params['n_features'], centers=params['target_clusters'] ,random_state=42)
        X = add_gaussian_noise(X, params['gaussian_noise_level'])
        X, y_true = add_structural_noise(X, y_true, params['structural_noise_level'])
        return X, y_true
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def test(config: Config=None):
    print("Getting data")
    X, y_true = get_dataset(config)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Initializing clusters")
    y_pred = {}
    
    pic = PathIntegralClustering(config.target_clusters, z=0.01, a=0.95, K=20)
    y_pred["AP"] = run_ap(X_scaled)
    y_pred["A-Link"] = runAC(X_scaled, config.target_clusters, method='average')
    y_pred["S-link"] = runAC(X_scaled, config.target_clusters, method='single')
    y_pred["C-link"] = runAC(X_scaled, config.target_clusters, method='complete')
    y_pred["Zell"] = zeta_function_clustering(X_scaled, config.target_clusters)
    y_pred["D-kernel"] = diffusion_kernel_clustering(X_scaled, config.target_clusters)
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

def evaluate_clustering(X, y_true, noise_level, n_clusters, noise_type="gaussian"):
    nmi_scores = {
        "AP": [],
        "A-Link": [],
        "S-link": [],
        "C-link": [],
        "Zell": [],
        "D-kernel": [],
        "PIC": []
    }
    
    for _ in range(20):  # Repeat for 20 times
        # Add noise to the data
        if noise_type == "gaussian":
            X_noisy = add_gaussian_noise(X, noise_level)
            y_true_def = y_true
        elif noise_type == "structural":
            X_noisy, y_true_def = add_structural_noise(X, y_true, noise_level)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_noisy)
    
        # Apply clustering
        y_pred = {}
        pic = PathIntegralClustering(n_clusters, z=0.01, a=0.95, K=20)
        y_pred["AP"] = run_ap(X_scaled)
        y_pred["A-Link"] = runAC(X_scaled, config.target_clusters, method='average')
        y_pred["S-link"] = runAC(X_scaled, config.target_clusters, method='single')
        y_pred["C-link"] = runAC(X_scaled, config.target_clusters, method='complete')
        y_pred["Zell"] = zeta_function_clustering(X_scaled, n_clusters)
        y_pred["D-kernel"] = diffusion_kernel_clustering(X_scaled, n_clusters)
        y_pred["PIC"] = pic.fit_predict(X_scaled)
        
        # Compute NMI score
        for key, val in y_pred.items():
            nmi_score = normalized_mutual_info_score(y_true_def, val)
            nmi_scores[key].append(nmi_score)
    
    # Return average NMI and standard deviation
    dist_nmi_scores = {}
    for key, val in nmi_scores.items():
        dist_nmi_scores[key] = np.mean(val), np.std(val)
        
    return dist_nmi_scores

def plot_results(results: dict, x_indices):
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
    plt.xlabel('Gaussian Noise Level')
    plt.ylabel('Score')
    plt.title('Performance vs. Gaussian Noise Level')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def run_noise_experiment(config: Config):
    print("Getting data")
    n_centers = config.target_clusters
    X, y_true = generate_synthetic(n_samples=config.n_samples, n_features=config.n_features, centers=n_centers,random_state=42)
    
    gaussian_noise_levels = [1,1.2,1.4,1.6,1.8]
    structural_noise_levels = [0,0.05,0.1,0.15,0.2,0.25]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    fig.suptitle('NMI Scores with Gaussian and Structural Noise', fontsize=16)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    ax = axes[0]
    results_gaussian = []
    file_gaussian = os.path.join(DATA_DIR,'gaussian_noise_comparison.pkl')
    if os.path.exists(file_gaussian):
        with open(file_gaussian, 'rb') as f:
            results_gaussian = pkl.load(f)
    else:
        for noise_level in tqdm.tqdm(gaussian_noise_levels, desc=f"Running gaussian noise experiment"):
            results_gaussian.append(evaluate_clustering(X_scaled, y_true, noise_level, n_centers, "gaussian"))
        with open(os.path.join(DATA_DIR,'gaussian_noise_comparison.pkl'), 'wb') as f:
            pkl.dump(results_gaussian, f)
        
    results_structural = []
    file_structural = os.path.join(DATA_DIR,'structural_noise_comparison.pkl')
    if os.path.exists(file_structural):
        with open(file_structural, 'rb') as f:
            results_gaussian = pkl.load(f)
    else:
        for noise_level in tqdm.tqdm(structural_noise_levels, desc=f"Running structural noise experiment"):
            results_structural.append(evaluate_clustering(X_scaled, y_true, noise_level, n_centers, "structural"))
    
        with open(os.path.join(DATA_DIR,'structural_noise_comparison.pkl'), 'wb') as f:
            pkl.dump(results_structural, f)
        
    plot_results(results_gaussian, gaussian_noise_levels)
    plot_results(results_structural, structural_noise_levels)

if __name__ == "__main__":
    config = Config(
        dataset_name="synthetic",
        n_samples=1000,
        n_features=10,
        target_clusters=10
    )
    run_noise_experiment(config)