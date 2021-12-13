import pandas as pd
import cloudfiles
import os

def read_json_file(cloudpath):
    if "://" not in cloudpath:
        cloudpath = "file://" + cloudpath
    folder,file = os.path.split(cloudpath)
    cf = cloudfiles.CloudFiles(folder)
    return cf.get_json(file)
    
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
