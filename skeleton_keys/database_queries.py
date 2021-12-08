import os
import allensdk.internal.core.lims_utilities as lu
from functools import partial


def default_query_engine():
    """Get Postgres query engine with environmental variable parameters"""

    return partial(
        lu.query,
        host=os.getenv("LIMS_HOST"),
        port=5432,
        database=os.getenv("LIMS_DBNAME"),
        user=os.getenv("LIMS_USER"),
        password=os.getenv("LIMS_PASSWORD")
    )


def query_for_swc_file(specimen_id, query_engine=None):
    if query_engine is None:
        query_engine = default_query_engine()

    query = f"""
    select
        f.filename as filename, f.storage_directory as storage_directory
    from neuron_reconstructions n
    join well_known_files f on n.id = f.attachable_id
    and n.specimen_id = {specimen_id}
    and n.manual
    and not n.superseded
    and f.well_known_file_type_id = 303941301
    """

    results = query_engine(query)
    if len(results) > 0:
        r = results[0]
        return os.path.join(r["storage_directory"], r["filename"])
    else:
        raise ValueError(f"No swc file path found for specimen ID {specimen_id}")


def swc_paths_from_database(specimen_ids):
    """Query the database for SWC paths

    Parameters
    ----------
    specimen_ids : list of ints
        List of specimen IDs

    Returns
    -------
    paths : dict
        Dictionary of paths keyed on specimen IDs
    """

    engine = default_query_engine()

    paths = {int(sp_id): query_for_swc_file(int(sp_id), engine)
        for sp_id in specimen_ids}
    return paths
