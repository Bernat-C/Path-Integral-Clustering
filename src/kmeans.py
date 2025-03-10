from sklearn.cluster import AffinityPropagation, AgglomerativeClustering

# 1. Affinity Propagation (AP)
def run_ap(X):
    ap = AffinityPropagation(random_state=42)
    ap_labels = ap.fit_predict(X)
    
    return ap_labels

# 2. Agglomerative Clustering with Average Linkage (A-Link) and S-Link
def runAC(X, method='average'):
    agglo_avg = AgglomerativeClustering(n_clusters=2, linkage=method) # average, single
    agglo_avg_labels = agglo_avg.fit_predict(X)
    return agglo_avg_labels
    

