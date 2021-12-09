import numpy as np
from ccf_streamlines.angle import vector_to_3d_affine_matrix, find_closest_streamline, determine_angle_between_streamline_and_plane


def slice_angle_tilt(pin_df, specimen_id, closest_surface_voxel_file, surface_paths_file):
    """ Estimate angle of slice from CCF pinning information

    Parameters
    ----------
    pin_df : DataFrame
        DataFrame with pinning query information
    closest_surface_voxel_file : str
        Path to closest surface voxel reference HDF5 file
    surface_paths_file : str
        Path to surface paths (streamlines) HDF5 file

    Returns
    -------
    predicted_slice_angle : float
        Estimated angle in radians
    """
    soma_coord = pin_df.set_index("specimen_id").loc[specimen_id, ["x", "y", "z"]].values.astype(float)
    tvr = pin_df.set_index("specimen_id").loc[specimen_id, ["tvr_00","tvr_01","tvr_02","tvr_03","tvr_04","tvr_05","tvr_06","tvr_07","tvr_08","tvr_09","tvr_10","tvr_11"]].values
    M = vector_to_3d_affine_matrix(tvr)
    streamline_coords = find_closest_streamline(soma_coord, closest_surface_voxel_file, surface_paths_file)
    predicted_slice_angle = determine_angle_between_streamline_and_plane(streamline_coords, M)

    return predicted_slice_angle * np.pi / 180.
