import pandas as pd
from importlib.resources import files
import json
from skeleton_keys.database_queries import (
    query_for_image_series_id,
    pia_wm_soma_from_database,
    layer_polygons_from_database
)

def load_default_layer_template():
    """ Load the default average cortical layer depth JSON file.

    Keys are strings representing cortical layers (e.g.'2/3','4'...)
    Values represent the cortical depth for the top (pia side) of a given layer

    Returns
    -------
    depths : dict
        Dictionary of distances to the pia (in microns) from the upper side of each layer
    """
    depth_file = files('skeleton_keys') / "test_files/mouse_me_and_met_avg_layer_depths.json"

    with open(depth_file, "r") as fn:
        depths = json.load(fn)
    return depths


def load_swc_as_dataframe(swc_file):
    """ Load a morphology SWC file into a pandas DataFrame.

    The dataframe contains the columns:
        - ID : node identifier
        - type : node type, which could be
            1 = soma
            2 = axon
            3 = basal dendrite (or generic dendrite)
            4 = apical dendrite
        - x : x-coordinate
        - y : y-coordinate
        - z : z-coordinate
        - r : radius
        - parent_id : identifier of parent node

    Parameters
    ----------
    swc_file : str
        File path of SWC file

    Returns
    -------
    df : DataFrame
        Dataframe with morphology information
    """
    return pd.read_table(
        swc_file,
        sep=" ",
        comment="#",
        names=["id", "type", "x", "y", "z", "r", "parent_id"],
    )


def save_lims_surface_and_layers_to_json(json_file, specimen_id):
    """ Save a set of databased surfaces and layers to a JSON file

    This function is only for users of the internal Allen Institute LIMS
    database.

    It saves the databased surface and layer drawings to a JSON file in the
    format expected by the ``skelekeys-layer-aligned-swc`` command line utility.

    Parameters
    ----------
    json_file : str
        Path to output JSON file
    specimen_id : int
        Specimen ID
    """

    # Query for image series ID
    imser_id = query_for_image_series_id(specimen_id)

    # Query for pia, white matter, and soma
    pia_surface, wm_surface, soma_drawing = pia_wm_soma_from_database(
        specimen_id, imser_id
    )

    print(pia_surface)
    print(pia_surface['path'])
    print(type(pia_surface['path']))

    # Query for layers
    layer_polygons = layer_polygons_from_database(imser_id)

    output_data = {
        "pia_path": pia_surface,
        "wm_path": wm_surface,
        "soma_path": soma_drawing,
        "layer_polygons": layer_polygons,
    }

    with open(json_file, "w") as out_f:
        json.dump(output_data, out_f)

