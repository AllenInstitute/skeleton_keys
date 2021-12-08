import sklearn.decomposition as decomposition


def calculate_pca_transforms_and_loadings(X, min_explained_variance=0.05):
    """ Get PCA-transformed depth profiles and loadings

    Parameters
    ----------
    X : array (n_samples, n_features)
        Input data of `n_samples` depth profiles with `n_features` elements
    min_explained_variance : float, default 0.05
        Minimum percentage of explained variance for component selection

    Returns
    -------
    X_new : array (n_samples, n_components)
        Transformed values
    loadings : array (n_components, n_features)
        PCA loadings that transform `X` to `X_new`
    """

    pca = decomposition.PCA()
    X_new = pca.fit_transform(X)
    keep_pcs = pca.explained_variance_ratio_ >= min_explained_variance

    return X_new[:, keep_pcs], pca.components_[keep_pcs, :]


def apply_loadings_to_profiles(X, loadings):
    """ Transform depth profiles using previously calculated loadings

    Parameters
    ----------
    X : array (n_samples, n_features)
        Input data of `n_samples` depth profiles with `n_features` elements
    loadings : array (n_components, n_features)
        PCA loadings that transform `X` to `X_new`

    Returns
    -------
    X_new : array (n_samples, n_components)
        Transformed values
    """

    X_new = X @ loadings.T
    return X_new
