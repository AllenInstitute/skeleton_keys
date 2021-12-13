import pandas as pd
import cloudfiles
import os
import io

def read_json_file(cloudpath):
    if "://" not in cloudpath:
        cloudpath = "file://" + cloudpath
    folder,file = os.path.split(cloudpath)
    cf = cloudfiles.CloudFiles(folder)
    return cf.get_json(file)

def read_bytes(path):
    if "://" not in path:
        path = "file://" + path

    cloudpath, file = os.path.split(path)
    cf = cloudfiles.CloudFiles(cloudpath)
    byt = io.BytesIO(cf.get(file))
    return byt 

def read_csv(path, **kwargs):

    byt = read_bytes(path)
    df = pd.read_csv(byt, **kwargs)
    return df

def write_dataframe_to_csv(df, path, **kwargs):
    if "://" not in path:
        path = "file://" + path
    cloudpath, file = os.path.split(path)
    cf = cloudfiles.CloudFiles(cloudpath)
    buffer = io.BytesIO()
    charset = 'utf-8'
    wrapper = io.TextIOWrapper(buffer, encoding=charset)
    df.to_csv(wrapper, **kwargs)
    buffer.seek(0)
    cf.put(file, buffer.getvalue(), content_type='application/x-csv')

    
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
