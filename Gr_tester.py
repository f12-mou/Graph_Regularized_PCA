# use only query cells (with spatial coords)
query = adata[adata.obs["sample"] == "diseased_1"].copy()

coords = query.obsm["spatial"]          # (n,2)
X = query.X                            # (n, genes) log-normalized

Z, U, X_mean = graph_regularized_pca(X, coords, n_components=10, k=8, lam=10.0)

query.obsm["X_grpca"] = Z  