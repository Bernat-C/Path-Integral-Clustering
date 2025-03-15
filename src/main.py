from sklearn.discriminant_analysis import StandardScaler
from data import generate_synthetic, load_bc_wisconsin, load_mnist, load_usps
from algorithms import run_ap, runAC, runDkernel, runZell
from sklearn.metrics import normalized_mutual_info_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from pic import PathIntegralClustering
from metrics import clustering_error
from visualize import visualize_clusters

def test():
    n_samples = 800
    n_features = 10
    target_clusters = 10
    
    print("Generating data")
    X, y_true = generate_synthetic(n_samples=n_samples, n_features=n_features, centers=target_clusters) # ,random_state=42)
    #X, y_true = load_usps()
    #X, y_true = load_mnist()
    #X, y_true = load_bc_wisconsin()
    
    scaler = StandardScaler()
    print(X.shape)
    X_scaled = scaler.fit_transform(X)
    
    print("Initializing clusters")
    y_pred = {}
    
    pic = PathIntegralClustering(target_clusters, z=0.01, a=0.95, K=20)
    y_pred["PIC"] = pic.fit_predict(X_scaled)
    y_pred["AP"] = run_ap(X_scaled)
    y_pred["A-Link"] = runAC(X_scaled, target_clusters, method='average')
    y_pred["S-link"] = runAC(X_scaled, target_clusters, method='single')
    y_pred["C-link"] = runAC(X_scaled, target_clusters, method='complete')
    #y_pred["D-link"] = runDkernel(X_scaled, target_clusters)
    
    visualize_clusters(X_scaled, y_pred["PIC"], title="Definitive clusters")
    print("=" * 120)
    print(f"{'Method':<10} | {'NMI':<10} | {'Silhouette':<12} | {'DB Score':<10} | {'Calinski-Harabasz':<18} | {'Clustering Error':<18}")
    print("-" * 120)

    for method in ["PIC", "AP", "A-Link", "S-link", "C-link"]:
        nmis = normalized_mutual_info_score(y_true, y_pred[method])
        silhouette = silhouette_score(X, y_pred[method])
        dbscore = davies_bouldin_score(X, y_pred[method])
        calinski = calinski_harabasz_score(X, y_pred[method])
        ce = clustering_error(y_true, y_pred[method])

        print(f"{method:<10} | {nmis:.4f}     | {silhouette:.4f}       | {dbscore:.4f}    | {calinski:.4f}               | {ce:.4f}")
        
    print("=" * 120)
    
if __name__ == "__main__":
    test()