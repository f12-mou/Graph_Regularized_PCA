import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.neighbors import NearestNeighbors

def build_spatial_knn_graph(coords, k=8, sigma=None):
    """
    coords: (n,2) array
    returns sparse weight matrix W (n,n)
    """
    n = coords.shape[0]
    nn = NearestNeighbors(n_neighbors=k+1).fit(coords)
    dists, idx = nn.kneighbors(coords)

    # remove self neighbor (first column)
    dists = dists[:, 1:]
    idx = idx[:, 1:]

    if sigma is None:
        # robust default: median neighbor distance
        sigma = np.median(dists)

    rows = np.repeat(np.arange(n), k)
    cols = idx.reshape(-1)
    w = np.exp(-(dists.reshape(-1) ** 2) / (2 * sigma**2))

    W = sp.coo_matrix((w, (rows, cols)), shape=(n, n))
    # symmetrize
    W = (W + W.T) / 2
    return W.tocsr()

def graph_laplacian(W, normalized=True):
    d = np.array(W.sum(axis=1)).flatten()
    D = sp.diags(d)

    if not normalized:
        return D - W

    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    Lsym = sp.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
    return Lsym

def graph_regularized_pca(X, coords, n_components=10, k=8, lam=10.0, normalized_laplacian=True):
    """
    X: (n_cells, n_genes) dense or sparse matrix
    coords: (n_cells, 2)
    Returns:
      Z: (n_cells, n_components) graph-smoothed PCs
      U: (n_genes, n_components) loadings
    """
    if sp.issparse(X):
        X = X.tocsr()
    else:
        X = np.asarray(X)

    # center genes (PCA assumes centered)
    X_mean = np.array(X.mean(axis=0)).ravel() if sp.issparse(X) else X.mean(axis=0)
    Xc = X - X_mean

    W = build_spatial_knn_graph(coords, k=k)
    L = graph_laplacian(W, normalized=normalized_laplacian)

    n = Xc.shape[0]
    A = (sp.eye(n) + lam * L).tocsr()  # (I + λL)

    # Apply (I + λL)^(-1) via linear solves (no explicit inverse)
    # Compute M = X^T A^{-1} X as an operator
    def Ainv_matmul(B):
        # Solve A Y = B for Y
        # B shape: (n, p)
        return spla.spsolve(A, B)

    # For efficiency: form M explicitly if genes not huge.
    # M = Xc^T (A^{-1} Xc)
    if sp.issparse(Xc):
        AX = Ainv_matmul(Xc.toarray())
        M = Xc.T @ AX
    else:
        AX = Ainv_matmul(Xc)
        M = Xc.T @ AX

    
    M = np.asarray(M)
    vals, vecs = np.linalg.eigh(M)
    U = vecs[:, np.argsort(vals)[::-1][:n_components]]  # top components

    # Z = A^{-1} X U
    XU = (Xc @ U) if not sp.issparse(Xc) else (Xc @ U)
    Z = Ainv_matmul(XU)

    return np.asarray(Z), np.asarray(U), X_mean