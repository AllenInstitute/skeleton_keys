import os
import logging
import numpy as np
import pandas as pd
try:
    import allensdk.internal.core.lims_utilities as lu
except ImportError:
    logging.warning('no lims internal installed')
from functools import partial
from neuron_morphology.marker import read_marker_file
from neuron_morphology.snap_polygons.types import ensure_path


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
    """Get an SWC file path for a specimen ID using the specified query engine"""
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


def query_for_image_series_id(specimen_id, query_engine=None):
    """Get an image series ID for a specimen ID using the specified query engine"""
    if query_engine is None:
        query_engine = default_query_engine()

    query = f"""
    select
        imser.id as image_series_id
    from specimens sp
    join specimens spp on spp.id = sp.parent_id
    join image_series imser on imser.specimen_id = spp.id
    where
        sp.id = {specimen_id}
    """

    results = query_engine(query)
    if len(results) > 0:
        return results[0]["image_series_id"]
    else:
        raise ValueError(f"No focal plane image series ID found for specimen ID {specimen_id}")


def query_for_cortical_surfaces(focal_plane_image_series_id, specimen_id=None, query_engine=None):
    """ Return the pia and white matter surface drawings for an image series (and optionally specimen ID)
    """
    if query_engine is None:
        query_engine = default_query_engine()

    query = f"""
        select distinct
            polygon.path as path,
            label.name as name,
            sc.resolution as resolution,
            bsp.biospecimen_id as biospecimen_id
        from specimens sp
        join specimens spp on spp.id = sp.parent_id
        join image_series imser on imser.specimen_id = spp.id
        join sub_images si on si.image_series_id = imser.id
        join images im on im.id = si.image_id
        join slides s on s.id=im.slide_id
        join scans sc on sc.slide_id=s.id
        join treatments tm on tm.id = im.treatment_id
        join avg_graphic_objects layer on layer.sub_image_id = si.id
        join avg_graphic_objects polygon on polygon.parent_id = layer.id
        join avg_group_labels label on label.id = layer.group_label_id
        JOIN biospecimen_polygons bsp ON bsp.polygon_id = polygon.id
        where
            imser.id = {focal_plane_image_series_id}
            and label.name in ('Pia', 'White Matter')
            and tm.name = 'Biocytin'
    """
    results = {}
    for item in query_engine(query):
        data = {
            "name": item["name"],
            "path": ensure_path(item["path"]),
            "resolution": item["resolution"],
            "biospecimen_id": item["biospecimen_id"]
        }
        if item["name"] not in results:
            results[item["name"]] = data
        elif specimen_id is not None and item["biospecimen_id"] == specimen_id:
            results[item["name"]] = data
        elif len(results[item["name"]]["path"]) < len(data["path"]):
            if specimen_id is not None and results[item["name"]]["biospecimen_id"] == specimen_id and item[
                "biospecimen_id"] != specimen_id:
                # don't replace if already have a match by ID (and this isn't also a match)
                pass
            else:
                results[item["name"]] = data

    return results["Pia"], results["White Matter"]


def query_for_soma_center(focal_plane_image_series_id, specimen_id, query_engine=None):
    """ Return the soma center coordinates for an image series ID and specimen ID
    """
    if query_engine is None:
        query_engine = default_query_engine()

    query = f"""
        select distinct
            polygon.path as path,
            label.name as name,
            sc.resolution as resolution,
            bsp.biospecimen_id as specimen_id
        from specimens sp
        join specimens spp on spp.id = sp.parent_id
        join image_series imser on imser.specimen_id = spp.id
        join sub_images si on si.image_series_id = imser.id
        join images im on im.id = si.image_id
        join slides s on s.id=im.slide_id
        join scans sc on sc.slide_id=s.id
        join treatments tm on tm.id = im.treatment_id
        join avg_graphic_objects layer on layer.sub_image_id = si.id
        join avg_graphic_objects polygon on polygon.parent_id = layer.id
        join avg_group_labels label on label.id = layer.group_label_id
        join biospecimen_polygons bsp on bsp.polygon_id = polygon.id
        where
            imser.id = {focal_plane_image_series_id}
            and label.name = 'Soma'
            and tm.name = 'Biocytin'
            and bsp.biospecimen_id = {specimen_id}
    """
    results = query_engine(query)
    if len(results) > 0:
        item = results[0]
        resolution = item["resolution"]
        path = ensure_path(item["path"])
        return {
            "name": item["name"],
            "path": path,
            "center": np.asarray(path).mean(axis=0).tolist(),
            "resolution": resolution,
        }
    else:
        raise ValueError(f"No soma coordinates found for specimen ID {specimen_id}")


def query_marker_file(specimen_id, query_engine=None):
    """ Return the marker file path for a specimen ID"""
    if query_engine is None:
        query_engine = default_query_engine()

    query = f"""
        select
            f.filename as filename,
            f.storage_directory as storage_directory
        from neuron_reconstructions n
        join well_known_files f on f.attachable_id = n.id
        and n.specimen_id = {specimen_id}
        and n.manual
        and not n.superseded
        and f.well_known_file_type_id = 486753749
    """
    results = query_engine(query)
    if len(results) > 0:
        return os.path.join(results[0]["storage_directory"], results[0]["filename"])
    else:
        return None


def query_cell_depth(specimen_id, query_engine=None):
    """ Cell depth from surface of slice for specimen"""
    if query_engine is None:
        query_engine = default_query_engine()

    query = f"""
        select
            sp.id as specimen_id,
            sp.cell_depth as cell_depth
        from specimens sp
        where sp.id = {specimen_id}
    """
    results = query_engine(query)
    if len(results) > 0:
        return results[0]["cell_depth"]
    else:
        raise ValueError(f"No cell depth found for specimen ID {specimen_id}")


def query_for_layer_polygons(focal_plane_image_series_id, query_engine=None):
    """ Get all layer polygons for this image series
    """
    if query_engine is None:
        query_engine = default_query_engine()

    query = f"""
        select distinct
            st.acronym as name,
            polygon.path as path,
            sc.resolution as resolution
        from specimens sp
        join specimens spp on spp.id = sp.parent_id
        join image_series imser on imser.specimen_id = spp.id
        join sub_images si on si.image_series_id = imser.id
        join images im on im.id = si.image_id
        join slides s on s.id=im.slide_id
        join scans sc on sc.slide_id=s.id
        join treatments tm on tm.id = im.treatment_id
        join avg_graphic_objects layer on layer.sub_image_id = si.id
        join avg_group_labels label on label.id = layer.group_label_id
        join avg_graphic_objects polygon on polygon.parent_id = layer.id
        join structures st on st.id = polygon.cortex_layer_id
        where
            imser.id = {focal_plane_image_series_id}
            and label.name in ('Cortical Layers')
            and tm.name = 'Biocytin' -- the polys are duplicated between 'Biocytin' and 'DAPI' images. Need only one of these
        """
    return [
        {
            "name": layer["name"],
            "path": ensure_path(layer["path"]),
            "resolution": layer["resolution"],
        }
        for layer in query_engine(query)
    ]


def query_pinning_info(project_codes=["T301", "T301x", "mIVSCC-MET"], query_engine=None):
    """Get the pinned CCF coordinates for a set of projects"""
    if query_engine is None:
        query_engine = default_query_engine()

    project_codes_str = ", ".join([f"'{s}'" for s in project_codes])
    query = f"""
        select distinct
            sp.id as specimen_id,
            csl.x as x,
            csl.y as y,
            csl.z as z,
            slice.id as slice_id,
            slab.id as slab_id,
            brain.id as brain_id,
            a3d.*
        from specimens sp
        join specimens slice on slice.id = sp.parent_id
        join specimens slab on slab.id = slice.parent_id
        join specimens brain on brain.id = slab.parent_id
        join alignment3ds a3d on slice.alignment3d_id = a3d.id
        join projects prj on prj.id = sp.project_id
        left join cell_soma_locations csl on sp.id = csl.specimen_id
        where prj.code in ({project_codes_str})
    """
    results = query_engine(query)
    return results


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


def pia_wm_soma_from_database(specimen_id, imser_id):
    """Get path strings for pia, wm, and soma drawing

    Parameters
    ----------
    specimen_id : int
        Specimen ID
    imser_id : int
        Image series ID

    Returns
    -------
    pia_path : str
        Pia drawing path
    wm_path : str
        White matter drawing path
    soma_path : str
        Soma drawing path
    resolution : float
        Resolution of drawings
    """
    engine = default_query_engine()

    pia_surface, wm_surface = query_for_cortical_surfaces(imser_id, specimen_id, query_engine=engine)
    soma_center = query_for_soma_center(imser_id, specimen_id, query_engine=engine)

    return pia_surface, wm_surface, soma_center


def shrinkage_factor_from_database(morph, specimen_id, cut_thickness=350.):
    """Determine shrinkage factor for morphology using database information

    Parameters
    ----------
    morph : Morphology
        Neuronal morphology
    cut_thickness : float, default 350.
    specimen_id : int
        Specimen ID
    cut_thickness : float, default 350.
        The cutting thickness (in microns) of the original slice. Used
        as an upper limit to the calculated thickness or as a fallback value
        if the thickness cannot be determined from the morphology and markers.

    Returns
    -------
    corrected_scale : float
        The factor to multiply the z-dimension by to adjust for shrinkage
    """
    engine = default_query_engine()
    cell_depth = query_cell_depth(specimen_id, engine)

    # special cases for missing LIMS entry
    special_cases = {992386952: 40.0,
                     738006528: 50.0,
                     848629140: 37.0,
                     848672037: 61.0,
                     701072075: 0}
    if cell_depth == 0 and specimen_id in special_cases.keys():
        cell_depth = special_cases[specimen_id]
    marker_file = query_marker_file(specimen_id, engine)

    if marker_file:
        markers = read_marker_file(marker_file)
    else:
        markers = []
    soma_marker = _identify_soma_marker(morph, markers)
    soma = morph.get_soma()
    if (soma_marker is not None) and (cell_depth != 0) and (cell_depth is not None):
        z_level = soma_marker["z"]
        fixed_depth = np.abs(soma["z"] - z_level)

        if np.allclose(fixed_depth, 0):
            logging.debug("zero depth found for {:d}".format(specimen_id))
            return np.nan

        scale = cell_depth / fixed_depth
        all_z = [c["z"] for c in morph.nodes()]
        max_z_extent = np.max(all_z) - np.min(all_z)
        min_slice_thickness = max_z_extent * scale

        if min_slice_thickness <= cut_thickness:
            corrected_scale = scale
        else:
            corrected_scale = cut_thickness / max_z_extent
    else:
        all_z = [c["z"] for c in morph.nodes()]
        max_z_extent = np.max(all_z) - np.min(all_z)
        corrected_scale = cut_thickness / max_z_extent

    return corrected_scale


def _identify_soma_marker(morph, markers, marker_tol=10.0):
    soma_markers = [m for m in markers if m["name"] == 30]  # 30 is the code for soma marker
    soma = morph.get_soma()
    if len(soma_markers) == 0:
        soma_marker = None
        for m in markers:
            if np.abs(soma["x"] - m["x"]) < marker_tol and np.abs(soma["y"] - m["y"]) < marker_tol:
                soma_marker = m
                break
        if soma_marker is None:
            logging.debug("no marker over soma found")
    else:
        soma_marker = soma_markers[0]
        # verify that soma marker is over the soma
        if np.abs(soma["x"] - soma_marker["x"]) > marker_tol or np.abs(soma["y"] - soma_marker["y"]) > marker_tol:
            soma_marker = None
            for m in markers:
                if np.abs(soma["x"] - m["x"]) < marker_tol and np.abs(soma["y"] - m["y"]) < marker_tol:
                    soma_marker = m
                    break
            if soma_marker is None:
                logging.debug("no marker over soma found")

    return soma_marker


def layer_polygons_from_database(image_series_id):
    """Obtain layer drawing polygons from database

    Parameters
    ----------
    image_series_id : int
        Image series ID

    Returns
    -------
    layers : list
        List of dictionaries containing layer `name`, `path`, and `resolution`
    """
    engine = default_query_engine()
    layer_polygons = query_for_layer_polygons(image_series_id, query_engine=engine)
    layer_polygons = [l for l in layer_polygons if len(l["path"]) >= 3]
    return layer_polygons


def determine_flip_switch(morph, specimen_id, revised_marker_file=None, marker_tol=10., return_info=False):
    """Multiplier for reversing slice angle. If rostral surface is on top, returns 1. Otherwise -1.

    Parameters
    ----------
    morph : Morphology
        Morphology object
    specimen_id : int
        Specimen ID
    revised_marker_file : str, default None
        Path to revised marker file (overrides marker file in database)
    marker_tol : float, default 10.
        Tolerance for soma marker identification (microns)
    return_info : bool, default False
        Whether to return information about flip determination

    Returns
    -------
        flip_status : int
            1 or -1 for toggling slice angle
        info : list
            information about how flip status was determined
    """

    engine = default_query_engine()
    marker_path = query_marker_file(specimen_id, engine)
    if revised_marker_file is not None:
        revised_marker_paths = pd.read_csv(revised_marker_file)
        if specimen_id in revised_marker_paths["specimen_id"].tolist():
            logging.debug("using revised markers for {:d}".format(specimen_id))
            marker_path = revised_marker_paths.set_index("specimen_id").at[specimen_id, "revised_marker_path"]

    if marker_path:
        markers = read_marker_file(marker_path)
    else:
        markers = []

    info_list = []

    soma_marker = _identify_soma_marker(morph, markers, marker_tol=marker_tol)
    flip_toggle = 1.  # Assume "flipped" ie rostral surface on top by default
    if soma_marker is not None:
        if (soma_marker["z"] - morph.get_soma()["z"]) > 0:
            print("cut surface is to the right!")
            info_list.append("cut surface appears to be to the 'right' from markers; adding a flip")
            flip_toggle *= -1
        else:
            info_list.append("cut surface appears to be to the 'left' as is typical")
    else:
        info_list.append("no soma marker to determine side of cut surface")

    flip_sql = f"""
        select distinct sp.id, sp.name, flip.name
        from specimens sp
        join specimens spp on spp.id = sp.parent_id
        join flipped_specimens flip on flip.id = spp.flipped_specimen_id
        where sp.id = {specimen_id}
    """

    results = engine(flip_sql)

    if len(results) == 0:
        info_list.append("no flip field found in LIMS; will flip by default")
    else:
        flip_result = results[0]['name']
        info_list.append("flip field has value of {}".format(flip_result))
        if flip_result == "flipped":
            info_list.append("adding a flip".format(flip_result))
        elif flip_result == "not flipped":
            info_list.append("not adding a flip".format(flip_result))
            flip_toggle *= -1
        else:
            info_list.append("adding a flip by default".format(flip_result))

    if return_info:
        return flip_toggle, info_list
    else:
        return flip_toggle
