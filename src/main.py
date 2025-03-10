from data import generate_synthetic, load_bc_wisconsin, load_mnist, load_usps
from sklearn.metrics import normalized_mutual_info_score
from pic import PathIntegralClustering
from metrics import clustering_error
from visualize import visualize_clusters

def get_instance_assignments(clusters):
    """ Transforms the cluster list, containing sets of instances belonging to each cluster to an array containing the assigned cluster index for each element in a sorted way.

    Args:
        clusters (List): array of arrays of cluster indices

    Returns:
        _type_: Cluster assignments
    """
    assignments = []
    
    for cluster_id, cluster in enumerate(clusters):
        for index in cluster:
            assignments.append((index, cluster_id))
    
    assignments.sort(key=lambda x: x[0])
    result = [cluster_id for index, cluster_id in assignments]
    
    return result

def test():
    n_samples = 500
    n_features = 10
    target_clusters = 10
    
    print("Generating data")
    data, y_true = generate_synthetic(n_samples=n_samples, n_features=n_features, centers=target_clusters)
    #data, y_true = load_usps()
    #data, y_true = load_bc_wisconsin()
    
    print("Initializing clusters")
    pic = PathIntegralClustering(target_clusters, z=0.01, a=0.95, K=20)
    C = pic.fit(data)

    visualize_clusters(data, C, range(len(C)), title="Definitive clusters")

    y_pred = get_instance_assignments(C)
    nmis = normalized_mutual_info_score(y_true,y_pred)
    ce = clustering_error(y_true,y_pred)
    print(f"Normalized mutual information score {nmis}")
    print(f"Clustering error {ce}")
    
if __name__ == "__main__":
    test()