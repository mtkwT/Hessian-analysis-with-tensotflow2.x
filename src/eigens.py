from scipy import linalg

def calculate_topK_eigens(hessian, k=2):
    """
    Because Hessian matrix is a real symmetric matrix,
    we should use "scypy.linalg.eigh" instead of "numpy.linalg.eig"
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eigh.html
    """
    eig_val, eig_vec = linalg.eigh(hessian, eigvals=(len(hessian)-k, len(hessian)-1))
    return eig_val, eig_vec
