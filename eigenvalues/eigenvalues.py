import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    if matrix is None or len(matrix) == 0:
        return None
        
    first_row_len = len(matrix[0]) if isinstance(matrix[0], (list, np.ndarray)) else 0
    if not all(isinstance(row, (list, np.ndarray)) and len(row) == first_row_len for row in matrix):
        return None
    matrix = np.array(matrix)
    if matrix.shape[0] != matrix.shape[1] or matrix.ndim !=2 :
        return None 
    eigen_val = np.linalg.eigvals(matrix)
    sort_indices = np.lexsort((eigen_val.imag,eigen_val.real))
    return eigen_val[sort_indices]