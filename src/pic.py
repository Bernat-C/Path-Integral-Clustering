import numpy as np
import heapq
import tqdm

from nearest_neighbour_init import cluster_init
from utils import k_nearest_neighbors, dist
from path_integral import compute_incremental_path_integral, compute_path_integral

class PathIntegralClustering:
    def __init__(self, target_clusters, z=0.01, a=0.95, K=20, verbose=False):
        self.target_clusters = target_clusters
        self.z = z
        self.a = a
        self.K = K
        self.verbose = verbose
        
    def _sigma_squared(self, X, knn3_dis,a):
        """Computes sigma squared for the PIC

        Args:
            X (np.array)
            knn3_dis (np.array): matrix containing distances to 3 nearest neighbours for each point in X
            a (float): free parameter

        Returns:
            sigma squared value
        """
        n = X.shape[0]
        
        squared_dists = np.sum(np.square(knn3_dis))
        den = 3*n*(-np.log(a))
            
        return squared_dists / den

    def _pairwise_similarity_matrix(self, X, knn, sigma2):
        """Computes the pairwise similarity matrix W

        Args:
            X (np.array)
            knn (np.array): matrix containing the indices of the nearest neigbours to each point in the dataset
            sigma2 (float): sigma squared value

        Returns:
            W: pairwise similarity matrix
        """
        n = len(X)
        W = np.zeros((n, n))  # Initialize the similarity matrix

        for i in tqdm.tqdm(range(n),desc="Computing matrix W", disable=not self.verbose):
            for j in knn[i]:  # Only compute for k-nearest neighbors
                W[i, j] = np.exp(-(dist(X[i], X[j]) ** 2 / sigma2))

        return W

    def _transition_probability_matrix(self, W):
        """ Computes the transition probability matrix

        Args:
            W (np.array): Pairwise similarity matrix

        Returns:
            P (np.array): Transition probability matrix
        """
        row_sums = np.sum(W, axis=1)  # Compute row sums (d_ii)
        
        # Avoid division by zero for isolated nodes
        np.place(row_sums, row_sums == 0, 1)

        # Compute D^(-1) * W using reciprocal for speed
        P = W * np.reciprocal(row_sums)[:, None]  # Element-wise multiplication

        return P

    def _create_digraph(self,X):
        """ Creates the initial digraph and computes its transition probability matrix

        Args:
            X (np.array): data

        Returns:
            W (np.array): transition probability matrix
        """
        if self.verbose:
            print(f"Creating Digraph")
        X = np.asarray(X)
        
        knn_ind, knn_dis = k_nearest_neighbors(X,self.K)
        knn3_ind, knn3_dis = k_nearest_neighbors(X,k=3)
        sigma2 = self._sigma_squared(X,knn3_dis,self.a)
            
        # Weighted adjacency matrix
        W = self._pairwise_similarity_matrix(X, knn_ind, sigma2)

        return W

    def _compute_affinity_single_link(self,D,Ca,Cb):
        """ Affinity using the single link method, used only to test

        Args:
            D (np.matrix): Distance matrix between the instances
            Ca (np.array): [[1],[2]...[n]]
            Cb (np.array): [[1],[2]...[n]]

        Returns:
            float: affinity
        """
        return -np.min([D[i, j] for i in Ca for j in Cb])

    def _compute_affinity(self,P, Ca, Cb): # Equation 3
        """ Affinity using path integral

        Args:
            Ca (np.array): [[1],[2]...[n]]
            Cb (np.array): [[1],[2]...[n]]

        Returns:
            float: affinity
        """
        Pa = P[np.ix_(Ca,Ca)]
        Pb = P[np.ix_(Cb,Cb)]

        S_Ca = compute_path_integral(Pa,self.z)
        #print(f"S_Ca: {S_Ca}")
        S_Cb = compute_path_integral(Pb,self.z)
        #print(f"S_Cb: {S_Cb}")
        S_Ca_given_CaUCb = compute_incremental_path_integral(Ca,Cb,P,self.z)
        #print(f"SCa given CaUb: {S_Ca_given_CaUCb}")
        S_Cb_given_CaUCb = compute_incremental_path_integral(Cb,Ca,P,self.z)
        #print(f"SCb given CaUb: {S_Cb_given_CaUCb}")

        result = (S_Ca_given_CaUCb - S_Ca) + (S_Cb_given_CaUCb - S_Cb)
        return result
    
    def _get_instance_assignments(self,clusters):
        """ Transforms the cluster list, containing sets of instances belonging to each cluster to an array containing the assigned cluster index for each element in a sorted way.

        Args:
            clusters (List): array of arrays of cluster indices

        Returns:
            Cluster assignments
        """
        assignments = []
        
        for cluster_id, cluster in enumerate(clusters):
            for index in cluster:
                assignments.append((index, cluster_id))
        
        assignments.sort(key=lambda x: x[0])
        result = [cluster_id for index, cluster_id in assignments]
        
        return result

    def run(self,X, C):
        """ Runs Agglomerative clustering via maximum incremental path integral.

        Args:
            X (np.array): set of n sample vectors X = {x1; x2;…; xn}
            C (np.array): cluster initializations
        """
        assert(self.K<len(X))
        
        W = self._create_digraph(X) # a can't be 0 nor 1 (inf)
        P = self._transition_probability_matrix(W)

        #D = cdist(X, X, metric="euclidean") # Used for trials using single-link instead of path integral
        
        nc = len(C)
        if nc < 2:
            return C # Nothing to merge
        
        max_heap = []
        for i in tqdm.tqdm(range(nc),desc="Computing initial cluster's affinities.", disable=not self.verbose):
            for j in range(i + 1, nc):
                affinity = self._compute_affinity(P, C[i], C[j])#compute_affinity_single_link(D,C[i],C[j])
                heapq.heappush(max_heap, (-affinity, i, j))  # Store negative affinity for max heap
                
        active_clusters = set(range(nc))
        
        if self.verbose:
            print("The clustering process has begun")
        for i in tqdm.tqdm(range(len(active_clusters), self.target_clusters, -1), desc="Clustering...", disable=not self.verbose):
            
            # Get the most similar pair
            while max_heap:
                _, i, j = heapq.heappop(max_heap)
                #print(f"{i} - {j} is a candidate.")
                if i in active_clusters and j in active_clusters:
                    # print(f"Cluster {i} {j} are the most affine.")
                    break  # Found valid cluster pair
            else:
                break
            
            merged = np.append(C[i], C[j])

            # Remove old elements and add merged element
            active_clusters.remove(i)
            active_clusters.remove(j)
            C.append(merged)
            new_idx = len(C) - 1
            active_clusters.add(new_idx)
            
            #print(f"Merged clusters {i} and {j} -> New cluster {new_idx}")

            # Update affinities with the new element
            for k in active_clusters - {new_idx}:  # Compute affinity with all previous elements excluding itself
                affinity =  self._compute_affinity(P, C[k], merged)#compute_affinity_single_link(D,C[k],merged)
                heapq.heappush(max_heap, (-affinity, k, new_idx))  # Push new affinities to the heap
        
        #print(f"Ended run with |C|:{len(C)} |AC|:{len(active_clusters)} with target clusters:{target_clusters}")
        return [C[i] for i in active_clusters]

    def fit_predict(self,X):
        C = cluster_init(X)
        #visualize_clusters(X, C, title="Clustering initialization")
        
        C = self.run(X,C)
        y = self._get_instance_assignments(C)
        
        return y