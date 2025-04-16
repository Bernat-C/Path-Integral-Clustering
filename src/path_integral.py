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


def compute_incremental_path_integral(C_a, C_b, P, z):
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
    C_a_union_C_b = sorted(np.append(C_a,C_b))
    P_CaUb = P[np.ix_(C_a_union_C_b, C_a_union_C_b)]

    C_size = len(P_CaUb)

    # Identity matrix of same shape as P_CaUb
    I = identity(C_size, format='csc')
    
    # Create indicator vector 1_Ca (same size as C_a_union_C_b)
    global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(C_a_union_C_b)}
    ones_Ca = np.zeros(C_size)
    for global_idx in C_a:
        local_idx = global_to_local[global_idx]  # Map global index to local index
        ones_Ca[local_idx] = 1  # Set elements corresponding to C_a as 1

    # Compute (I - Z * P_CaUb)^(-1)
    A = I - z * csc_matrix(P_CaUb)
    
    y = spsolve(A, ones_Ca)
    
    # Compute path integral using the formula
    S_Ca_given_CaUb = (1 / len(C_a)**2) * np.dot(ones_Ca.T, y)
    
    return S_Ca_given_CaUb

def compute_incremental_path_integral_inefficient(Ca, Cb, P, z):
    """
    Computes the incremental path integral S_{Ca | Ca ∪ Cb}.

    Parameters:
    Ca : List or set of indices representing cluster Ca.
    Cb : List or set of indices representing cluster Cb.
    P  : 2D numpy array, transition probability matrix.
    z  : Scalar value.

    Returns:
    float : Incremental path integral value.
    """
    # Combine indices for the submatrix extraction
    C_union = sorted(np.append(Ca,Cb)) # Sorted to maintain order

    # Create the submatrix P_Ca∪Cb
    P_C_union = P[np.ix_(C_union, C_union)]

    # Identity matrix of the same size
    I = np.eye(len(C_union))

    # Compute the inverse (I - z * P_C_union)
    M_inv = np.linalg.inv(I - z * P_C_union)

    # Indicator vector for Ca (1 for indices in Ca, 0 otherwise)
    ones_Ca = np.array([1 if i in Ca else 0 for i in C_union]).reshape(-1, 1)

    # Compute the incremental path integral
    S = (1 / (len(Ca) ** 2)) * ones_Ca.T @ M_inv @ ones_Ca

    return float(S)  # Convert to scalar