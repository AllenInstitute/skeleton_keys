.. _database_queries:

.. module:: skeleton_keys.database_queries

Database Queries (:mod:`skeleton_keys.database_queries`)
========================================================

These functions are used to interface with the private Allen Institute for
Brain Science LIMS database. Users external to the Allen Institute will generally
not have a use for these functions.


.. currentmodule:: skeleton_keys


Queries with default engine
---------------------------

.. autosummary::
    :toctree: generated/

    database_queries.swc_paths_from_database
    database_queries.pia_wm_soma_from_database
    database_queries.shrinkage_factor_from_database
    database_queries.layer_polygons_from_database
    database_queries.determine_flip_switch


Queries with specified engine
-----------------------------

.. autosummary::
    :toctree: generated/

    database_queries.query_for_swc_file
    database_queries.query_for_image_series_id
    database_queries.query_for_cortical_surfaces
    database_queries.query_for_layer_polygons
    database_queries.query_for_soma_center
    database_queries.query_marker_file
    database_queries.query_cell_depth
    database_queries.query_pinning_info


Query Engine
------------

.. autosummary::
    :toctree: generated/

    database_queries.default_query_engine
