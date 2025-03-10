from sklearn.discriminant_analysis import StandardScaler
from data import generate_synthetic, load_bc_wisconsin, load_mnist, load_usps
from kmeans import run_ap, runAC
from sklearn.metrics import normalized_mutual_info_score
from pic import PathIntegralClustering
from metrics import clustering_error
from visualize import visualize_clusters

def test():
    n_samples = 500
    n_features = 10
    target_clusters = 10
    
    print("Generating data")
    #X, y_true = generate_synthetic(n_samples=n_samples, n_features=n_features, centers=target_clusters ,random_state=42)
    #X, y_true = load_usps()
    X, y_true = load_mnist()
    #X, y_true = load_bc_wisconsin()
    
    scaler = StandardScaler()
    print(X.shape)
    X_scaled = scaler.fit_transform(X)
    
    print("Initializing clusters")
    y_pred = {}
    
    pic = PathIntegralClustering(target_clusters, z=0.01, a=0.95, K=20)
    y_pred["PIC"] = pic.fit_predict(X_scaled)
    y_pred["AP"] = run_ap(X_scaled)
    y_pred["A-Link"] = runAC(X_scaled, method='average')
    y_pred["S-link"] = runAC(X_scaled, method='single')
    y_pred["C-link"] = runAC(X_scaled, method='complete')
    
    visualize_clusters(X_scaled, y_pred["PIC"], title="Definitive clusters")
    for method in ["PIC","AP","A-Link","S-link","C-link"]:
        print(f" ---- {method} ----")
        nmis = normalized_mutual_info_score(y_true,y_pred[method])
        ce = clustering_error(y_true,y_pred[method])
        print(f"Normalized mutual information score {nmis:.4f}")
        print(f"Clustering error {ce:.4f}")
    
if __name__ == "__main__":
    test()