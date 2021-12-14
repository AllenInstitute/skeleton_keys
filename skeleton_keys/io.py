import pandas as pd
import cloudfiles
import os
import io
import re


def fix_local_cloudpath(cloudpath):
    if "://" not in cloudpath:
        cloudpath = "file://" + cloudpath
    return cloudpath

def get_path_files(cloudpath, regex_template=None):
    cloudpath = fix_local_cloudpath(cloudpath)
    cf = cloudfiles.CloudFiles(cloudpath)
    if regex_template is None:
        filenames = [f"{cloudpath}/{fn}" for fn in cf.list()]
    else:
        filenames = [f"{cloudpath}/{fn}" for fn in cf.list() if re.search(regex_template, fn) is not None]
    return filenames

def read_json_file(cloudpath):
    path = fix_local_cloudpath(cloudpath)
    folder, file = os.path.split(cloudpath)
    cf = cloudfiles.CloudFiles(folder)
    return cf.get_json(file)


def read_bytes(path):
    path = fix_local_cloudpath(path)
    cloudpath, file = os.path.split(path)
    cf = cloudfiles.CloudFiles(cloudpath)
    byt = io.BytesIO(cf.get(file))
    return byt


def read_csv(path, **kwargs):
    byt = read_bytes(path)
    df = pd.read_csv(byt, **kwargs)
    return df


def write_dataframe_to_csv(df, path, **kwargs):
    path = fix_local_cloudpath(path)
    cloudpath, file = os.path.split(path)
    cf = cloudfiles.CloudFiles(cloudpath)
    buffer = io.BytesIO()
    charset = "utf-8"
    wrapper = io.TextIOWrapper(buffer, encoding=charset)
    df.to_csv(wrapper, **kwargs)
    wrapper.seek(0)
    cf.put(file, buffer.getvalue(), content_type="text/csv")


def write_json(data, path):
    path = fix_local_cloudpath(path)
    cloudpath, file = os.path.split(path)
    cf = cloudfiles.CloudFiles(cloudpath)
    cf.put_json(file, data)


def load_swc_as_dataframe(swc_file):
    """Load a morphology SWC file into a pandas DataFrame.

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
    return read_csv(
        swc_file,
        sep=" ",
        comment="#",
        names=["id", "type", "x", "y", "z", "r", "parent_id"],
    )
