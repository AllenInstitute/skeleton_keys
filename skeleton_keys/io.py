import pandas as pd
import pkg_resources
import json

def load_default_layer_template():
    """
    Load the default average cortical layer depth json. Keys are strings representing cortical layers (e.g.'2/3','4'...)
    Values represent the cortical depth for the top (pia side) of a given layer

    :return:
    depths: Dictionary

    """
    depth_file = pkg_resources.resource_filename(__name__, 'test_files/avg_layer_depths.json')
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
