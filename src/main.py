import os
import re
import ast
import tqdm
import numpy as np
import pandas as pd
import pickle as pkl
from pathlib import Path
from matplotlib import pyplot as plt

from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import normalized_mutual_info_score

from config import Config, load_configs
from data import generate_synthetic, load_bc_wisconsin, load_mnist, load_usps, add_gaussian_noise, add_structural_noise
from algorithms import run_ap, runAC, zeta_function_clustering, diffusion_kernel_clustering
from pic import PathIntegralClustering
from metrics import clustering_error
from visualize import visualize_clusters, plot_noise_results

DATA_DIR =  Path(__file__).parent / '..' / 'data'
PLOTS_DIR = DATA_DIR / 'plots'

def get_dataset(config: Config):
    """ Get the dataset based on the configuration """
    if config.dataset_name == "usps":
        return load_usps()
    elif config.dataset_name == "mnist":
        return load_mnist()
    elif config.dataset_name == "bc_wisconsin":
        return load_bc_wisconsin()
    elif config.dataset_name == "blobs" or config.dataset_name == "moons" or config.dataset_name == "circles":
        return generate_synthetic(n_samples=config.n_samples, n_features=config.n_features, centers=config.target_clusters, ds_type=config.dataset_name)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset_name}")

def test(config: Config):
    """ Test the clustering algorithms on the specified dataset """
    
    print("Getting data")
    X, y_true = get_dataset(config)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Initializing clusters")
    y_pred = {}
    
    y_pred["A-Link"] = runAC(X_scaled, config.target_clusters, method='average')
    pic = PathIntegralClustering(config.target_clusters, z=0.01, a=0.95, K=20)
    y_pred["AP"] = run_ap(X_scaled)
    y_pred["S-link"] = runAC(X_scaled, config.target_clusters, method='single')
    y_pred["C-link"] = runAC(X_scaled, config.target_clusters, method='complete')
    y_pred["Zell"] = zeta_function_clustering(X_scaled, config.target_clusters)
    y_pred["D-kernel"] = diffusion_kernel_clustering(X_scaled, config.target_clusters)
    y_pred["PIC"] = pic.fit_predict(X_scaled)
    
    visualize_clusters(X_scaled, y_pred["PIC"], title="Definitive clusters", save_path=False)#PLOTS_DIR / f"{config.name}_pic.png")
    
    results = []
    for method in ["PIC","AP","A-Link","S-link","C-link","Zell","D-kernel"]:
        print(f" #################### {method} ####################")
        
        nmis = normalized_mutual_info_score(y_true, y_pred[method])
        ce = clustering_error(y_true, y_pred[method])
        silhouette = silhouette_score(X_scaled, y_pred[method]) 
        davies_bouldin = davies_bouldin_score(X_scaled, y_pred[method])
        calinski_harabasz = calinski_harabasz_score(X_scaled, y_pred[method])
        
        print(f"Normalized mutual information score: {nmis:.4f}")
        print(f"Clustering error: {ce:.4f}")
        print(f"Silhouette score: {silhouette:.4f}")
        print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
        print(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")
        
        results.append({
            'Method': method,
            'NMI': nmis,
            'Clustering Error': ce,
            'Silhouette Score': silhouette,
            'Davies-Bouldin Index': davies_bouldin,
            'Calinski-Harabasz Index': calinski_harabasz
        })
        
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(DATA_DIR,f'{config.name}_results.csv'), index=False)

def evaluate_clustering(X, y_true, noise_level, n_clusters, noise_type="gaussian"):
    """Evaluate clustering algorithms on the dataset with added noise."""
    algorithms = {
        "AP": lambda X: run_ap(X),
        "A-Link": lambda X: runAC(X, n_clusters, method='average'),
        "S-link": lambda X: runAC(X, n_clusters, method='single'),
        "C-link": lambda X: runAC(X, n_clusters, method='complete'),
        "Zell": lambda X: zeta_function_clustering(X, n_clusters),
        "D-kernel": lambda X: diffusion_kernel_clustering(X, n_clusters),
        "PIC": lambda X: PathIntegralClustering(n_clusters, z=0.01, a=0.95, K=20).fit_predict(X)
    }

    nmi_scores = {alg: [] for alg in algorithms}

    for it in tqdm.tqdm(range(20)):
        # Add noise to the data
        if noise_type == "gaussian":
            X_noisy = add_gaussian_noise(X, noise_level)
            y_true_def = y_true
        elif noise_type == "structural":
            X_noisy, y_true_def = add_structural_noise(X, y_true, noise_level)
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")

        X_scaled = StandardScaler().fit_transform(X_noisy)

        # Apply clustering algorithms
        y_preds = {alg: func(X_scaled) for alg, func in algorithms.items()}

        for alg, y_pred in y_preds.items():
            if it == 0:
                visualize_clusters(
                    X_scaled, y_pred,
                    title="Definitive clusters",
                    save_path=PLOTS_DIR / f"{config.name}_{config.dataset_name}_{alg}_{noise_type}_{noise_level}.png"
                )
            nmi = normalized_mutual_info_score(y_true_def, y_pred)
            nmi_scores[alg].append(nmi)

    # Return average NMI and standard deviation
    return {alg: (np.mean(scores), np.std(scores)) for alg, scores in nmi_scores.items()}

def run_single_noise_type(noise_type, noise_levels, exp_type, X_scaled, y_true, n_centers):
    """ Run experiments for a single noise type """
    results = []
    csv_file = os.path.join(DATA_DIR, f'{noise_type}_noise_{exp_type}.csv')
    df_result = pd.read_csv(csv_file) if os.path.exists(csv_file) else pd.DataFrame()

    for noise_level in (pbar := tqdm.tqdm(noise_levels, desc=f"Running {noise_type} noise experiment")):
        if 'noise_level' in df_result.columns and noise_level in df_result['noise_level'].values:
            res = df_result[df_result['noise_level'] == noise_level].iloc[0].to_dict()
            res_treated = {
                key: el if key == "noise_level" else tuple(
                    np.float64(x) for x in ast.literal_eval(
                        re.sub(r'np.float64\((.*?)\)', r'\1', el)
                    )
                )
                for key, el in res.items()
            }
            results.append(res_treated)
            continue

        pbar.set_postfix({'noise_level': noise_level})
        result = evaluate_clustering(X_scaled, y_true, noise_level, n_centers, noise_type)
        result['noise_level'] = noise_level
        results.append(result)

        pd.DataFrame(results).to_csv(csv_file, index=False)

    for r in results:
        r.pop("noise_level", None)

    plot_noise_results(
        results, noise_levels, 
        noise_type=noise_type.capitalize(), 
        save_path=PLOTS_DIR / f"results_{noise_type}_noise_{exp_type}.png"
    )
        
def run_noise_experiment(config: Config):
    """ Run the noise experiment on synthetic datasets """
    print("Getting data")
    n_centers = config.target_clusters
    exp_type = config.dataset_name

    # Generate and scale data
    X, y_true = generate_synthetic(
        n_samples=config.n_samples, 
        n_features=config.n_features, 
        centers=n_centers,
        ds_type=exp_type
    )
    n_centers = len(set(y_true))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define noise configs
    noise_configs = {
        "gaussian": [1, 1.2, 1.4, 1.6, 1.8],
        "structural": [0, 0.05, 0.1, 0.15, 0.2, 0.25]
    }
    
    for noise_type, noise_levels in noise_configs.items():
        run_single_noise_type(noise_type, noise_levels, exp_type, X_scaled, y_true, n_centers)

if __name__ == "__main__":
    # Single experiment
    config = Config(
        name=f"synthetic",
        dataset_name="blobs",
        n_samples=500,
        n_features=2,
        target_clusters=10
    )
        
    test(config)
    
    # Synthetic dataset noise experiment
    # for ds in ["moons","blobs","circles"]:
    #     config = Config(
    #         name=f"synthetic_noise_{ds}",
    #         dataset_name=ds,
    #         n_samples=1000,
    #         n_features=2,
    #         target_clusters=10
    #     )
    #     run_noise_experiment(config)
    
    # # Experiment with real datasets
    # configs = load_configs()
    # for config in configs:
    #     test(config)