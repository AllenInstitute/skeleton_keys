.. _cmds:

.. module:: skeleton_keys.cmds

Command Line Utilities (:mod:`skeleton_keys.cmds`)
========================================================

These command line utility scripts are designed to be used together to create
a morphology feature processing pipeline.

You can access information about the arguments for each script by
passing the ``--help`` option on the command line.

.. currentmodule:: skeleton_keys.cmds


``skelekeys-layer-aligned-swc``
-------------------------------

Script to generate a layer-aligned SWC file.

.. autosummary::
    :toctree: generated/

    layer_aligned_swc.LayerAlignedSwcSchema


``skelekeys-upright-corrected-swc``
-----------------------------------

Script to generate an upright, slice angle-corrected SWC file (but without
layer alignment).

.. autosummary::
    :toctree: generated/

    upright_corrected_swc.UprightCorrectedSwcSchema
    upright_corrected_swc.soma_distance_from_pia


``skelekeys-profiles-from-swcs``
--------------------------------

Script to calculate layer-aligned depth profile histograms from a set of SWC files.

.. autosummary::
    :toctree: generated/

    depth_profiles_from_aligned_swcs.ProfilesFromAlignedSwcsParameters


``skelekeys-calc-histogram-loadings``
-------------------------------------

Script to calculate the PCA loadings for a set of layer-aligned depth profile
histograms.

.. autosummary::
    :toctree: generated/

    calc_histogram_loadings.CalcHistogramLoadingsParameters


``skelekeys-morph-features``
----------------------------

Script to calculate morphological features for a set of SWC files.

.. autosummary::
    :toctree: generated/

    process_morphology_features.ProcessMorphologyFeaturesParameters
    process_morphology_features.soma_locations
    process_morphology_features.select_and_convert_depth_columns
    process_morphology_features.analyze_depth_profiles
    process_morphology_features.specimen_morph_features

``skelekeys-postprocess-features``
----------------------------------

Script to perform postprocessing on the long-form morphology feature files (convert
to wide format and perform z-score normalization).

When this script normalizes the values, it normalizes all the values with the same
feature name to the same scale. This means that, for example, the feature `extent`
will be normalized in both `x` and `y` dimensions (i.e., width and height) with
the same scale. The intention is to avoid overemphasizing differences in particular
dimensions where slight differences may not be relevant.

.. autosummary::
    :toctree: generated/

    postprocess_features.PostprocessFeaturesParameters


``skelekeys-layer-aligned-coords``
----------------------------------

Script to layer-aligned coordinates from a general CSV file.

.. autosummary::
    :toctree: generated/

    layer_aligned_coords.LayerAlignedCoordsSchema


``skelekeys-profiles-from-coords``
----------------------------------

Script to calculate layer-aligned depth profile histograms from a CSV with
layer-aligned coordinates.

.. autosummary::
    :toctree: generated/

    depth_profiles_from_aligned_coords.ProfilesFromAlignedCoordsParameters
