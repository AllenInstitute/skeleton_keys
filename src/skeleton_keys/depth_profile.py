import logging
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


def overlap_between_compartments(compartment_a_df, compartment_b_df):
    """ Calculate overlap fractions between two types of compartments

    Parameters
    ----------
    compartment_a_df : DataFrame
        DataFrame of depth profiles for first compartment type
    compartment_b_df : Dataframe
        DataFrame of depth profiles for second compartment type

    Returns
    -------
    overlap_df : DataFrame
        Overlap fractions as DataFrame
    """
    result = {}
    for sp_id in compartment_a_df.index:
        a_values = compartment_a_df.loc[sp_id, :].values
        b_values = compartment_b_df.loc[sp_id, :].values

        a_total = a_values.sum()

        if a_total == 0:
            result[sp_id] = {
                "frac_above": 0.,
                "frac_intersect": 0.,
                "frac_below": 0.,
            }
            continue

        intersections = a_values.astype(bool) & b_values.astype(bool)

        # earlier values in the array are shallower/upper
        upper_limit = np.argmax(intersections).astype(int)

        # reverse the array to find the last intersection
        lower_limit = len(intersections) - np.argmax(intersections[::-1]).astype(int)

        if intersections.sum() == 0:
            # completely non-overlapping
            # so determine which is above the other
            a_median_index = np.median(np.flatnonzero(a_values > 0))
            b_median_index = np.median(np.flatnonzero(b_values > 0))

            if a_median_index < b_median_index:
                result[sp_id] = {
                    "frac_above": 1.,
                    "frac_intersect": 0.,
                    "frac_below": 0.,
                }
            else:
                result[sp_id] = {
                    "frac_above": 0.,
                    "frac_intersect": 0.,
                    "frac_below": 1.,
                }
        else:
            result[sp_id] = {
                "frac_above": a_values[:upper_limit].sum() / a_total,
                "frac_intersect": a_values[upper_limit:lower_limit + 1].sum() / a_total,
                "frac_below": a_values[lower_limit + 1:].sum() / a_total,
            }
    result_df = pd.DataFrame.from_dict(result, orient="index")
    return result_df
