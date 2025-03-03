import numpy as np
from path_integral import compute_incremental_path_integral, compute_path_integral
from main import compute_affinity

# Test cases
def test_compute_affinity():
    np.random.seed(42)

    # Generate a random 5x5 affinity matrix (symmetric, non-negative)
    P = np.random.rand(5, 5)
    P = (P + P.T) / 2  # Make it symmetric

    Ca = np.array([0, 1])  # First cluster indices
    Cb = np.array([2, 3])  # Second cluster indices
    z = 0.5  # Arbitrary weight parameter

    # Compute affinity
    affinity = compute_affinity(P, Ca, Cb, z)

    # Print results
    print("Affinity matrix P:\n", P)
    print("\nCluster Ca indices:", Ca)
    print("Cluster Cb indices:", Cb)
    print("\nComputed Affinity:", affinity)

    # Assertions for correctness
    assert isinstance(affinity, float), "Affinity should be a float"
    assert affinity >= 0, "Affinity should be non-negative (assuming positive matrix elements)"
    
test_compute_affinity()
# P = np.array([[0.1, 0.5, 0.4], 
#               [0.3, 0.4, 0.3], 
#               [0.2, 0.3, 0.5]])  # Example transition probability matrix

# C_a = [0]  # Cluster A contains node 0
# C_b = [1, 2]  # Cluster B contains nodes 1 and 2

# P_a = P[np.ix_(C_a, C_a)]
# Sa = compute_path_integral(P_a, z=1.0)
# result = compute_incremental_path_integral(C_a, C_b, P, z=1.0)
# print("Path Integral of a:", Sa)
# print("Incremental Path Integral:", result)