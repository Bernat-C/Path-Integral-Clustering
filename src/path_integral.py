import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import identity, csc_matrix

def compute_path_integral(Pc, z):
    """
    Compute the path integral structural descriptor S_c.
    
    Parameters:
    Pc (numpy.ndarray): Submatrix of transition matrix P for cluster C.
    z (float): Scaling factor for P_C.
    
    Returns:
    float: Computed value of S_c.
    """
    C_size = Pc.shape[0]  # Number of elements in the cluster
    I = identity(C_size, format='csc')  # Identity matrix
    
    # (I - zP_C)
    A = I - z * csc_matrix(Pc)
        
    # Equation 10 Solve (I - zP_C) y = 1
    y = spsolve(A, np.ones(C_size))
    
    # Equation 11
    S_c = (1 / C_size**2) * np.sum(y)
    
    return S_c


def compute_incremental_path_integral(C_a, C_b, P, z=1.0):
    """
    Compute the incremental path integral as per equation (12).
    
    Parameters:
    - C_a: List of node indices in cluster C_a.
    - C_b: List of node indices in cluster C_b.
    - P: Transition probability matrix (numpy array of shape (n, n)).
    - Z: Scalar parameter in the equation (defaults to 1.0).
    
    Returns:
    - Incremental path integral value.
    """
    # Combine clusters to get the submatrix P_CaUb
    C_a_union_C_b = np.array(C_a + C_b)
    C_size = len(C_a_union_C_b)
    P_CaUb = P[np.ix_(C_a_union_C_b, C_a_union_C_b)]  # Extract submatrix

    # Identity matrix of same shape as P_CaUb
    I = identity(C_size, format='csc')

    # Create indicator vector 1_Ca (same size as C_a_union_C_b)
    ones_Ca = np.zeros(len(C_a_union_C_b))
    ones_Ca[:len(C_a)] = 1  # Set first |C_a| elements to 1
    
    # Compute (I - Z * P_CaUb)^(-1)
    A = I - z * csc_matrix(P_CaUb)
    y = spsolve(A, ones_Ca)

    # Compute path integral using the formula
    S_Ca_given_CaUb = (1 / len(C_a) ** 2) * np.dot(ones_Ca.T, y)

    return S_Ca_given_CaUb
