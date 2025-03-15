from sklearn.cluster import AffinityPropagation, AgglomerativeClustering
import pygkernels as pgk

# 1. Affinity Propagation (AP)
def run_ap(X):
    ap = AffinityPropagation(random_state=42)
    ap_labels = ap.fit_predict(X)
    
    return ap_labels

# 2. Agglomerative Clustering with Average Linkage (A-Link) and Single Linkage (S-Link)
def runAC(X, n_clusters, method='average'):
    agglo_avg = AgglomerativeClustering(n_clusters=n_clusters, linkage=method) # average, single
    agglo_avg_labels = agglo_avg.fit_predict(X)
    return agglo_avg_labels
    
# 3. Zeta function based clustering, Zell
def runZell(X):
    pass

# 4. Difusion based kernel, D-kernel
def runDkernel(X, n_clusters):
    kernel = pgk.cluster.HeatKernel()
    K = kernel.compute(X)

    # Apply Kernel K-Means clustering
    kmeans = pgk.measure.KernelKMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(K)

    return labels