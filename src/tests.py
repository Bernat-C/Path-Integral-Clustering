import numpy as np
from path_integral import compute_incremental_path_integral, compute_path_integral

P = np.array([[0.1, 0.5, 0.4], 
              [0.3, 0.4, 0.3], 
              [0.2, 0.3, 0.5]])  # Example transition probability matrix

C_a = [0]  # Cluster A contains node 0
C_b = [1, 2]  # Cluster B contains nodes 1 and 2

P_a = P[np.ix_(C_a, C_a)]
Sa = compute_path_integral(P_a, z=1.0)
result = compute_incremental_path_integral(C_a, C_b, P, z=1.0)
print("Path Integral of a:", Sa)
print("Incremental Path Integral:", result)