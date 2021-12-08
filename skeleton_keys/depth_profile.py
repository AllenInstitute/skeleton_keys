import numpy as np
import pandas as pd
import sklearn.decomposition as decomposition
import scipy.stats as stats



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


def earthmover_distance_between_compartments(compartment_a_df, compartment_b_df):
    """ Calculate earthmover distance between two types of compartments

    Parameters
    ----------
    compartment_a_df : DataFrame
        DataFrame of depth profiles for first compartment type
    compartment_b_df : Dataframe
        DataFrame of depth profiles for second compartment type

    Returns
    -------
    emd_df : DataFrame
        Earthmover distances as DataFrame
    """

    locations = np.arange(compartment_a_df.shape[1])

    emd_dict = {}
    for sp_id in compartment_a_df.index:
        a_values = compartment_a_df.loc[sp_id, :].values
        b_values = compartment_b_df.loc[sp_id, :].values

        if (a_values.sum() == 0) or (b_values.sum() == 0):
            # if either type is not present, set to sum of the other
            emd = a_values.sum() + b_values.sum()
        else:
            emd = stats.wasserstein_distance(locations, locations, a_values, b_values)
        emd_dict[sp_id] = emd

    df = pd.DataFrame.from_dict(emd_dict, orient='index')
    df.columns = ['emd']
    return df

