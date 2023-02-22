import logging
import vtk
import numpy as np
from ccf_streamlines.angle import find_closest_streamline
from ccf_streamlines.coordinates import coordinates_to_voxels
from neuron_morphology.transforms.affine_transform import (
    rotation_from_angle, affine_from_transform_translation, affine_from_translation, AffineTransform
)
from neuron_morphology.swc_io import morphology_from_swc
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from skimage.measure import find_contours
from scipy.interpolate import interpn
from scipy.spatial.distance import euclidean
from scipy.spatial.transform import Rotation
from vtk.util.numpy_support import vtk_to_numpy
from scipy import ndimage
from scipy.spatial import KDTree


def check_coord_in_structure(coordinate, structure_id, atlas_volume, closest_surface_voxel_file, surface_paths_file,
                             tree):
    """
    Check if a given ccf coordinate (microns) is located in a certain CCF structure

    :param coordinate: array (1x3)
        Location of coordinate to check (microns)
    :param structure_id: int
        ccf structure id to check if the input coordinate is in. E.g. if structure_id=315,
        this script would check if the input coordinate is located in iso-cortex
    :param atlas_volume: 3d atlas array
        Region-annotated CCF atlas volume
    :param closest_surface_voxel_file: str
        Closest surface voxel reference HDF5 file path for angle calculation
    :param surface_paths_file: str
        Surface paths (streamlines) HDF5 file path for slice angle calculation:
    :param tree: structure tree
        from allensdk package:

    :return:
    out_of_cortex : bool
        True if the input coordinate is out of cortex, otherwise is False
    nearest_cortex_coord : array/None
        Nearest iso-cortex coordinate if the input coordinate is out of cortex, otherwise is None

    """
    # get structure of input voxel
    voxel = coordinates_to_voxels(coordinate.reshape(1, 3))[0]
    voxel_struct_id = atlas_volume[voxel[0], voxel[1], voxel[2]]

    # find all descendant structures of input structure_id
    structure_ids = tree.descendant_ids([structure_id])[0]

    nearest_cortex_coord = None
    out_of_cortex = False
    if voxel_struct_id not in structure_ids:
        # the structure does not contain our coordinate of interest, find the nearest voxel
        nearest_cortex_coord, nearest_cortex_voxel = find_neaerst_isocortex_structure(voxel,
                                                                                      atlas_volume,
                                                                                      structure_ids,
                                                                                      closest_surface_voxel_file,
                                                                                      surface_paths_file)
        nearest_cortex_coord = np.array([nearest_cortex_coord[0], nearest_cortex_coord[1], nearest_cortex_coord[2]])

        out_of_cortex = True

    return out_of_cortex, nearest_cortex_coord


def find_neaerst_isocortex_structure(out_of_cortex_voxel, atlas_volume, isocortex_ids,
                                     closest_surface_voxel_file, surface_paths_file, atlas_resolution=10.):
    """
    Given an out of cortex voxel, this will find the nearest iso-cortex voxel.

    :param out_of_cortex_voxel: array (1x3)
        voxel location of point that is out of cortex
    :param atlas_volume: 3d atlas array
        Region-annotated CCF atlas volume
    :param isocortex_ids: list
        list of iso-cortex structure ids (ints)
    :param closest_surface_voxel_file: str
        Closest surface voxel reference HDF5 file path for angle calculation
    :param surface_paths_file: str
        Surface paths (streamlines) HDF5 file path for slice angle calculation
    :param atlas_resolution: float, default 10.
        Voxel size of atlas volume (microns)

    :return:
    soma_coords : array
        closest isocortex point (micron)
    new_soma_voxel : array
        closest isocortex point (voxel)
    """

    # Mask isocortex structures and erode one step
    isocortex_mask = np.isin(atlas_volume, isocortex_ids)
    struct = ndimage.generate_binary_structure(3, 3)
    eroded_cortex_mask = ndimage.morphology.binary_erosion(isocortex_mask, structure=struct).astype(int)

    isocortex_perimeter = np.subtract(isocortex_mask, eroded_cortex_mask)
    perim_coords = np.array(np.where(isocortex_perimeter != 0)).T

    # Find nearest isocortex voxel
    perim_kd_tree = KDTree(perim_coords)
    dists, inds = perim_kd_tree.query(out_of_cortex_voxel.reshape(1, 3), k=1)
    perim_coord_index = inds[0]

    closest_cortex_voxel = perim_coords[perim_coord_index]
    closest_cortex_coord = atlas_resolution * closest_cortex_voxel

    streamline = find_closest_streamline(closest_cortex_coord, closest_surface_voxel_file, surface_paths_file)

    # go one up to prevent this coord from existing directly on the white matter boundary
    closest_cortex_coord = streamline[-2]
    new_soma_voxel = closest_cortex_coord / atlas_resolution

    return closest_cortex_coord, new_soma_voxel.astype(int)


def rotate_morphology_for_drawings_by_angle(morphology, angle_for_atlas_slice, base_orientation='coronal',
                                            additional_data=None):
    """ Rotate a whole-brain morphology to line up with angled atlas image

    In the CCF, x is anterior/posterior, y is dorsal/ventral, and z is left/right

    In a near-coronal slice, though, the first dimension is (primarily) dorsal/ventral and the
    second is left/right.

    In a near-parasaggital slice, the first dimension is dorsal/ventral and the second is
    anterior/posterior.

    This function both tilts the morphology to line up
    with the atlas slice and swaps the coordinates so that the dimensions are
    correctly matched up.

    Parameters
    ----------
    morphology : Morphology
        Full brain morphology, aligned to CCF (micron scale)
    angle_for_atlas_slice : float
        Angle of atlas slice to align to streamline (radians)
    base_orientation : str, default 'coronal'
        Initial slice orientation - either 'coronal' or 'parasagittal'
    additional_data : nx3 array,
        Additional coordinates to apply the same rotation to. Specifically useful when
        a cells soma is in the white matter and the nearest iso-cortex voxel needs to be
        rotated to line up with the angled atlas image.
    Returns
    -------
    morphology : Morphology
        Rotated morphology with soma in original location

    """

    if base_orientation == 'coronal':
        rot_axis = 2
        coordinate_swap_matrix = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ]).T
    elif base_orientation == 'parasagittal':
        rot_axis = 1
        coordinate_swap_matrix = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ]).T
    else:
        raise ValueError("base_orientation must be 'coronal' or 'parasagittal'")

    rotation_tilt_matrix = rotation_from_angle(-angle_for_atlas_slice, axis=rot_axis)
    morphology,additional_data = _rotate_morphology_around_soma(morphology, rotation_tilt_matrix, additional_data)
    morphology,additional_data = _swap_morphology_coordinates(morphology, coordinate_swap_matrix, additional_data)

    return morphology, additional_data


def rotate_morphology_for_drawings_by_rotation(morphology, q, additional_data=None):
    """ Rotate a whole-brain morphology to line up with angled atlas image

    In the CCF, x is anterior/posterior, y is dorsal/ventral, and z is left/right

    In a near-coronal slice, though, the first dimension is (primarily) dorsal/ventral and the
    second is left/right.

    In a near-parasaggital slice, the first dimension is dorsal/ventral and the second is
    anterior/posterior.

    This function both tilts the morphology to line up
    with the atlas slice and swaps the coordinates so that the dimensions are
    correctly matched up.

    Parameters
    ----------
    morphology : Morphology
        Full brain morphology, aligned to CCF (micron scale)
    q : Rotation
        SciPy rotation object that generated the angled atlas image

    Returns
    -------
    morphology : Morphology
        Rotated morphology with soma in original location

    """

    M = q.inv().as_matrix()
    morphology, additional_data = _rotate_morphology_around_soma(morphology, M, additional_data)

    coordinate_swap_matrix = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ])
    morphology = _swap_morphology_coordinates(morphology, coordinate_swap_matrix, additional_data)

    return morphology


def align_morphology_to_drawings(morphology, atlas_slice, atlas_resolution=10.):
    """ Move the soma to the center of a set of drawings

    Parameters
    ----------
    morphology : Morphology
        Full brain morphology, aligned to CCF (micron scale)
    atlas_slice : 2D array
        Angled atlas slice with CCF region IDs as values
    atlas_resolution : float, default 10
        Voxel size of atlas volume (microns)

    Returns
    -------
    aligned_morphology : Morphology
        Translated morphology
    """
    # Align morphology to center of drawings
    soma_morph = morphology.get_soma()
    translation = np.array([
        -soma_morph['x'] + atlas_resolution * atlas_slice.shape[0] / 2,
        -soma_morph["y"] + atlas_resolution * atlas_slice.shape[1] / 2,
        -soma_morph["z"]
    ])
    translation_affine = affine_from_translation(translation)
    T_translate = AffineTransform(translation_affine)
    T_translate.transform_morphology(morphology)

    return morphology


def _rotate_morphology_around_soma(morphology, M, additional_data=None):
    """Rotates morphology (and additional data) around its soma but maintains original position"""

    # center on soma before rotation
    soma_morph = morphology.get_soma()
    translation_to_origin = np.array([-soma_morph['x'], -soma_morph["y"], -soma_morph["z"]])
    translation_back_to_soma_location = -translation_to_origin

    translation_affine = affine_from_translation(translation_to_origin)
    T_translate = AffineTransform(translation_affine)
    T_translate.transform_morphology(morphology)
    if additional_data is not None:
        additional_data = T_translate.transform(additional_data)

    # Rotate
    rotation_tilt_affine = affine_from_transform_translation(transform=M)
    T_rotate = AffineTransform(rotation_tilt_affine)
    T_rotate.transform_morphology(morphology)
    if additional_data is not None:
        additional_data = T_rotate.transform(additional_data)

    # Move back
    translation_affine = affine_from_translation(translation_back_to_soma_location)
    T_translate = AffineTransform(translation_affine)
    T_translate.transform_morphology(morphology)
    if additional_data is not None:
        additional_data = T_translate.transform(additional_data)

    return morphology, additional_data


def _swap_morphology_coordinates(morphology, M, additional_data = None):
    """Use matrix to switch coordinates"""

    # Swap coordinates
    swap_affine = affine_from_transform_translation(transform=M)
    T_swap = AffineTransform(swap_affine)
    T_swap.transform_morphology(morphology)
    if additional_data is not None:
        additional_data = T_swap.transform(additional_data)

    return morphology,additional_data


def _angle_between_streamline_and_plane(streamline_coords, norm_vec):
    norm_unit = norm_vec / euclidean(norm_vec, [0, 0, 0])

    streamline_wm = streamline_coords[-1, :]
    streamline_pia = streamline_coords[0, :]
    streamline_unit = (streamline_pia - streamline_wm) / euclidean(streamline_pia, streamline_wm)

    angle_with_norm = np.arctan2(
        np.linalg.norm(np.cross(norm_unit, streamline_unit)),
        np.dot(norm_unit, streamline_unit))

    # Take complement and convert from radians to degrees
    return 90. - (angle_with_norm * 180. / np.pi)


def angled_atlas_slice_for_morph(morph, atlas_volume,
        closest_surface_voxel_reference_file, surface_paths_file,
        base_orientation='auto', closest_cortex_node=None,
        atlas_resolution=10., return_angle=True):
    """ Create an angled atlas slice lined up with a cell's streamline

    Parameters
    ----------
    morph : Morphology
        Full brain morphology, aligned to CCF (micron scale)
    atlas_volume : 3D array
        Region-annotated CCF atlas volume
    closest_surface_voxel_reference_file : str
        Closest surface voxel reference HDF5 file path for angle calculation
    surface_paths_file : str
        Surface paths (streamlines) HDF5 file path for slice angle calculation
    base_orientation : str, default 'auto'
        Initial slice orientation - either 'auto' (pick option with smaller tilt),
        'coronal' or 'parasagittal'
    atlas_resolution : float, default 10
        Voxel size of atlas volume (microns)
    return_angle : bool, default True
        Whether to return the slice angle and orientation along with the slice
    closest_cortex_node : array, default None
        If the morphology's soma is outside the cortex, this value is required to
        find the nearest streamline. It represents the closest isocortex node to
        the cells soma.
    Returns
    -------
    atlas_slice : 2D array
        Angled atlas slice with CCF region IDs as values
    angle_rad : float
        Angle of slice from original plane (radians)
    slice_orientation : str
        Orientation of slice (for auto selection; otherwise matches base_orientation)
    """

    if closest_cortex_node is None:
        morph_soma = morph.get_soma()
        soma_coords = np.array([morph_soma['x'], morph_soma['y'], morph_soma['z']])
    else:
        soma_coords = closest_cortex_node

    # Find streamline for cell
    morph_streamline = find_closest_streamline(
        soma_coords, closest_surface_voxel_reference_file, surface_paths_file)

    if base_orientation == 'auto':
        coronal_norm_vec = [1, 0, 0]
        coronal_angle_deg = _angle_between_streamline_and_plane(morph_streamline, coronal_norm_vec)
        para_norm_vec = [0, 0, -1]
        para_angle_deg = _angle_between_streamline_and_plane(morph_streamline, para_norm_vec)
        if np.abs(para_angle_deg) < np.abs(coronal_angle_deg):
            norm_vec = para_norm_vec
            base_orientation = 'parasagittal'
            logging.info(f"Auto selecting parasagittal orientation ({para_angle_deg:.2f} angle vs {coronal_angle_deg:.2f} for coronal)")
        else:
            norm_vec = coronal_norm_vec
            base_orientation = 'coronal'
            logging.info(f"Auto selecting coronal orientation ({coronal_angle_deg:.2f} angle vs {para_angle_deg:.2f} for parasagittal)")
    elif base_orientation == 'coronal':
        norm_vec = [1, 0, 0]
    elif base_orientation == 'parasagittal':
        norm_vec = [0, 0, -1]
    else:
        raise ValueError("base_orientation must be 'auto', 'coronal', or 'parasagittal'")

    # Get angle between coronal section and cell's streamline
    angle_deg = _angle_between_streamline_and_plane(morph_streamline, norm_vec)
    logging.info(f"Tilting by {angle_deg:.2f} degrees from {base_orientation} plane")

    angle_rad = angle_deg * np.pi / 180.

    soma_coords_for_atlas = soma_coords / atlas_resolution

    if base_orientation == 'coronal':
        # Set up a grid representing a coronal section at the level of the cell body
        slice_grid = np.meshgrid(
            soma_coords_for_atlas[0],
            np.arange(0, atlas_volume.shape[1], 1),
            np.arange(0, atlas_volume.shape[2], 1),
        )
        rot_axis = 2
        reshape_size = (atlas_volume.shape[1], atlas_volume.shape[2])
    elif base_orientation == 'parasagittal':
        slice_grid = np.meshgrid(
            np.arange(0, atlas_volume.shape[0], 1),
            np.arange(0, atlas_volume.shape[1], 1),
            soma_coords_for_atlas[2],
        )
        rot_axis = 0
        reshape_size = (atlas_volume.shape[1], atlas_volume.shape[0])
    mesh_coords = np.array([slice_grid[0].flatten(), slice_grid[1].flatten(), slice_grid[2].flatten()])

    # Rotate the mesh
    M = rotation_from_angle(angle_rad, axis=rot_axis)
    rot_mesh_coords = (M @ (mesh_coords - soma_coords_for_atlas[:, np.newaxis])).T + soma_coords_for_atlas

    # Get an interpolated slice through the mesh. Use nearest method since
    # averaging ID values has no meaning.
    atlas_points = (
        np.arange(atlas_volume.shape[0], dtype=float),
        np.arange(atlas_volume.shape[1], dtype=float),
        np.arange(atlas_volume.shape[2], dtype=float)
    )
    atlas_slice = interpn(
        atlas_points,
        atlas_volume,
        rot_mesh_coords,
        method='nearest').astype(int)

    atlas_slice = atlas_slice.reshape(reshape_size)

    if return_angle:
        return atlas_slice, angle_rad, base_orientation
    else:
        return atlas_slice


def min_curvature_atlas_slice_for_morph(morph, atlas_volume,
        closest_surface_voxel_reference_file, surface_paths_file,
        pia_curvature_surface_file, wm_curvature_surface_file,
        closest_cortex_node=None, atlas_resolution=10.):
    """ Create an angled atlas slice lined up with a cell's streamline

    The soma location of the morphology will be in the center of the returned slice.

    Parameters
    ----------
    morph : Morphology
        Full brain morphology, aligned to CCF (micron scale)
    atlas_volume : 3D array
        Region-annotated CCF atlas volume
    closest_surface_voxel_reference_file : str
        Closest surface voxel reference HDF5 file path for angle calculation
    surface_paths_file : str
        Surface paths (streamlines) HDF5 file path for slice angle calculation
    pia_curvature_surface_file : str
        VTP file path for pia surface curvature values
    wm_curvature_surface_file : str
        VTP file path for white matter surface curvature values
    atlas_resolution : float, default 10
        Voxel size of atlas volume (microns)

    Returns
    -------
    atlas_slice : 2D array
        Angled atlas slice with CCF region IDs as values
    rot : Rotation object
        Rotation object that produced the slice
    """
    if closest_cortex_node is None:
        morph_soma = morph.get_soma()
        soma_coords = np.array([morph_soma['x'], morph_soma['y'], morph_soma['z']])
    else:
        soma_coords = closest_cortex_node


    # check if on right hemisphere
    on_right = False
    if soma_coords[2] > (atlas_resolution * atlas_volume.shape[2] / 2):
        on_right = True

        # Blank out opposite hemisphere
        atlas_volume = atlas_volume.copy()
        atlas_volume[:, :, :atlas_volume.shape[2] // 2] = 0
    else:
        # Blank out opposite hemisphere
        atlas_volume = atlas_volume.copy()
        atlas_volume[:, :, atlas_volume.shape[2] // 2:] = 0

    # Find streamline for cell (coordinates in um)

    if not on_right:
        morph_streamline = find_closest_streamline(
            soma_coords, closest_surface_voxel_reference_file, surface_paths_file)
    else:
        flipped_soma_coords = soma_coords.copy()
        flipped_soma_coords[2] = atlas_resolution * atlas_volume.shape[2] - soma_coords[2]
        morph_streamline = find_closest_streamline(
            flipped_soma_coords, closest_surface_voxel_reference_file, surface_paths_file)

    # Convert to mm
    top_streamline_coords_mm = morph_streamline[0, :] / 1000.
    bottom_streamline_coords_mm = morph_streamline[-1, :] / 1000.

    # Load the surfaces with curvature information
    # (surface coordinates are in mm)
    pia_reader = vtk.vtkXMLPolyDataReader()
    pia_reader.SetFileName(pia_curvature_surface_file)
    pia_reader.Update()
    pia_surf = pia_reader.GetOutput()

    wm_reader = vtk.vtkXMLPolyDataReader()
    wm_reader.SetFileName(wm_curvature_surface_file)
    wm_reader.Update()
    wm_surf = wm_reader.GetOutput()

    # Get the vectors in direction of maximum curvature (ie those normal to minimum curvature)
    pia_t1_arr = vtk_to_numpy(pia_surf.GetCellData().GetArray("t_1"))
    wm_t1_arr = vtk_to_numpy(wm_surf.GetCellData().GetArray("t_1"))

    # Get the max (kappa1) & min (kappa2) curvature values
    pia_kappa1_arr = vtk_to_numpy(pia_surf.GetCellData().GetArray("kappa_1"))
    wm_kappa1_arr = vtk_to_numpy(wm_surf.GetCellData().GetArray("kappa_1"))
    pia_kappa2_arr = vtk_to_numpy(pia_surf.GetCellData().GetArray("kappa_2"))
    wm_kappa2_arr = vtk_to_numpy(wm_surf.GetCellData().GetArray("kappa_2"))

    pia_delta_kappa_arr = pia_kappa1_arr - pia_kappa2_arr
    wm_delta_kappa_arr = wm_kappa1_arr - wm_kappa2_arr

    # Find the values of curvature nearest top & bottom of streamline
    pia_point_id = pia_surf.FindPoint(top_streamline_coords_mm)
    wm_point_id = wm_surf.FindPoint(bottom_streamline_coords_mm)
    pia_cell_id_list = vtk.vtkIdList()
    wm_cell_id_list = vtk.vtkIdList()
    pia_surf.GetPointCells(pia_point_id, pia_cell_id_list)
    wm_surf.GetPointCells(wm_point_id, wm_cell_id_list)

    # Just use the first cell found
    pia_cell_idx = pia_cell_id_list.GetId(0)
    wm_cell_idx = wm_cell_id_list.GetId(0)

    # Pick the end that has bigger potential increase in curvature if changed from its minimum direction
    if pia_delta_kappa_arr[pia_cell_idx] >= wm_delta_kappa_arr[wm_cell_idx]:
        norm_vec = pia_t1_arr[pia_cell_idx]
    else:
        norm_vec = wm_t1_arr[wm_cell_idx]

    if on_right:
        # Flip the z-direction
        norm_vec[2] = -norm_vec[2]

    # Set up the reference mesh (parasagittal orientation) centered on the soma
    soma_coords_for_atlas = soma_coords / atlas_resolution
    ref_norm = np.array([0, 0, 1])
    slice_grid = np.meshgrid(
        np.arange(0, atlas_volume.shape[0], 1) + soma_coords_for_atlas[0] - atlas_volume.shape[0] / 2,
        np.arange(0, atlas_volume.shape[1], 1) + soma_coords_for_atlas[1] - atlas_volume.shape[1] / 2,
        soma_coords_for_atlas[2],
    )
    reshape_size = (atlas_volume.shape[1], atlas_volume.shape[0])
    mesh_coords = np.array([
        slice_grid[0].flatten(),
        slice_grid[1].flatten(),
        slice_grid[2].flatten()
    ])

    # Calculate the rotation by quaternion
    a = np.cross(ref_norm, norm_vec)
    w = 1 + np.dot(ref_norm, norm_vec)
    q = Rotation.from_quat([a[0], a[1], a[2], w])

    # Rotate the mesh
    M = q.as_matrix()
    rot_mesh_coords = (M @ (mesh_coords - soma_coords_for_atlas[:, np.newaxis])).T + soma_coords_for_atlas

    # Get an interpolated slice through the mesh. Use nearest method since
    # averaging ID values has no meaning.
    atlas_points = (
        np.arange(atlas_volume.shape[0], dtype=float),
        np.arange(atlas_volume.shape[1], dtype=float),
        np.arange(atlas_volume.shape[2], dtype=float)
    )
    atlas_slice = interpn(
        atlas_points,
        atlas_volume,
        rot_mesh_coords,
        method='nearest',
        fill_value=0,
        bounds_error=False,
    ).astype(int)

    atlas_slice = atlas_slice.reshape(reshape_size)

    return atlas_slice, q


def select_structures_of_interest(atlas_image, structure_list, tree=None):
    """ Select particular structures of interest and mask out others

    Parameters
    ----------
    atlas_image : 2D array
        Atlas image with CCF region IDs as values
    structure_list : list of str
        List of region acronyms to keep (including all descendants) in returned image
    tree : StructureTree, optional
        Structure tree (from allensdk package)

    Returns
    -------
    new_atlas_image : 2D array
        Atlas image with non-selected regions masked out
    """
    if tree is None:
        reference_space_key = 'annotation/ccf_2017'
        resolution = 10
        rspc = ReferenceSpaceCache(resolution, reference_space_key, manifest='manifest.json')
        tree = rspc.get_structure_tree(structure_graph_id=1)

    structure_list_ids = [d["id"] for d in tree.get_structures_by_acronym(structure_list)]
    descendants = tree.descendant_ids(structure_list_ids)
    combined_descendants = []
    for d in descendants:
        combined_descendants += d
    mask = np.isin(atlas_image, combined_descendants)
    new_atlas_image = atlas_image.copy()
    new_atlas_image[~mask] = 0
    return new_atlas_image


def merge_atlas_layers(atlas_image, tree=None):
    """ Combine regions associated with the same cortical layers

    Parameters
    ----------
    atlas_image : 2D array
        Atlas image with CCF region IDs as values
    tree : StructureTree, optional
        Structure tree (from allensdk package)

    Returns
    -------
    remapped_atlas_image : 2D array
        Atlas image with layers merged
    """

    if tree is None:
        reference_space_key = 'annotation/ccf_2017'
        resolution = 10
        rspc = ReferenceSpaceCache(resolution, reference_space_key, manifest='manifest.json')
        tree = rspc.get_structure_tree(structure_graph_id=1)

    layer_structure_sets = {
        "Isocortex layer 1": 667481440,
        "Isocortex layer 2/3": 667481441,
        "Isocortex layer 4": 667481445,
        "Isocortex layer 5": 667481446,
        "Isocortex layer 6a": 667481449,
        "Isocortex layer 6b": 667481450,
    }

    ent_layer_structures = {
        'Isocortex layer 1': [1121, 526, 259],
        'Isocortex layer 2/3': [20, 999, 715, 764, 543, 468, 508, 324, 52, 664, 371],
        'Isocortex layer 4': [92, 312, 712, 419],
        'Isocortex layer 5': [139, 727],
        'Isocortex layer 6a': [28, 743],
        'Isocortex layer 6b': [60]
    }

    uniq_structs, inv = np.unique(atlas_image, return_inverse=True)

    structure_layer_mapping = {}
    for iso_cortex_lyr, struct_set_id in layer_structure_sets.items():
        structures_of_set = [d['id'] for d in tree.get_structures_by_set_id([struct_set_id])]
        for ent_lyr_struct_id in ent_layer_structures[iso_cortex_lyr]:
            structures_of_set.append(ent_lyr_struct_id)

        for struct_id in uniq_structs:
            if struct_id in structures_of_set:
                structure_layer_mapping[struct_id] = struct_set_id
    structure_layer_mapping[0] = 0

    remapped_atlas_image = np.array([structure_layer_mapping[x] for x in uniq_structs])[inv].reshape(atlas_image.shape)
    return remapped_atlas_image


def remove_other_hemisphere(atlas_image, soma_coords, ccf_dimension=(1320, 800, 1140), resolution=10):
    """ Mask out all structures on the opposite side of the brain

    Parameters
    ----------
    atlas_image : 2D array
        Atlas image (angled coronal slice)
    soma_coords : (3, ) array
        Soma coordinates in the CCF (microns)
    ccf_dimensions : tuple, default (1320, 800, 1140)
        Size of CCF volume (voxels)
    resolution : float, default 10
        Voxel size (microns)

    Returns
    -------
    new_atlas_image : 2D array
        Atlas image with opposite side hemisphere structures masked out
    """

    removed_atlas_image = atlas_image.copy()
    center_of_brain = ccf_dimension[2] // 2
    if soma_coords[2] < center_of_brain * resolution:
        removed_atlas_image[:, removed_atlas_image.shape[1] // 2:] = 0
    else:
        removed_atlas_image[:, :removed_atlas_image.shape[1] // 2] = 0
    return removed_atlas_image


def find_layer_outlines(atlas_image, soma_coords, resolution=10, min_contour_pts=10):
    """ Identify boundaries of each layer from a 2D atlas image

    Atlas image must contain the structure set IDs of the cortical layers

    Parameters
    ----------
    atlas_image : 2D array
        Atlas image
    soma_coords : (3, ) array
        Soma coordinates aligned to the drawings
    resolution : float, default 10
        Voxel size (microns)
    min_contour_pts : int, default 10
        Minimum number of points to consider contour as an option

    Returns
    -------
    boundaries : dict
        Dictionary with layer names as keys and boundary coordinates as values
    """
    layer_structure_sets = {
        "Isocortex layer 1": 667481440,
        "Isocortex layer 2/3": 667481441,
        "Isocortex layer 4": 667481445,
        "Isocortex layer 5": 667481446,
        "Isocortex layer 6a": 667481449,
        "Isocortex layer 6b": 667481450,
    }

    boundaries = {}
    for name, layer_id in layer_structure_sets.items():
        region_raster = np.zeros_like(atlas_image).astype(int)
        region_raster[atlas_image == layer_id] = 1
        contours = find_contours(region_raster, level=0.5)

        if len(contours) == 0:
            # No contours found - don't add
            pass
        elif len(contours) == 1:
            boundaries[name] = contours[0]
        else:
            # Find the contour closest to the soma
            min_dist = np.inf
            for c in contours:
                if c.shape[0] < min_contour_pts:
                    # Skip very small contours
                    continue
                centroid = c.mean(axis=0) * resolution
                dist = euclidean(soma_coords[:2], centroid)
                if dist < min_dist:
                    boundaries[name] = c
                    min_dist = dist
    return boundaries


def find_structures_morphology_occupies(swc_file, atlas_volume, tree,
                                        structures_of_interest=["Isocortex", "Entorhinal area"]):
    """
    Given a ccf-registered morphology this will return the overlap between the structures that cell occupies
    and a list of regions of interest.
    :param swc_file: str
        path to swc file
    :param atlas_volume: 3D array
        Region-annotated CCF atlas volume
    :param tree: structure tree
        from allensdk package
    :param structures_of_interest: list
        list of structures used to mask cell. E.g. if cell has node(s) in Pons but structures_of_interest
        list does not contain Pons or any Pons ontological ancestors, Pons will not be returned in
        final_structures

    :return: list
        list of structures the cell occupies that are descendants of structures_of_interest
    """
    morphology = morphology_from_swc(swc_file)

    id_by_acronym = tree.get_id_acronym_map()
    acronym_by_id = {v: k for k, v in id_by_acronym.items()}
    id_by_name = {v: k for k, v in tree.get_name_map().items()}

    # Find all valid descendant structures from the structures of interest list
    approved_acronyms = set()
    for roi_struct_name in structures_of_interest:
        roi_id = id_by_name[roi_struct_name]
        all_roi_descendants = tree.descendants([roi_id])[0]
        for struct_dict in all_roi_descendants:
            approved_acronyms.add(struct_dict['acronym'])

    # Find the voxel annotations for the coordinates the cell occupies
    ccf_node_coordinates = np.array([[n['x'], n['y'], n['z']] for n in morphology.nodes()])
    voxels = coordinates_to_voxels(ccf_node_coordinates)
    structure_list_ids = set([atlas_volume[v[0], v[1], v[2]] for v in voxels])
    structure_list = set([acronym_by_id[s_id] for s_id in structure_list_ids if s_id != 0])

    # Overlap between approved valid structures and the structures the cell occupies
    cells_valid_structures = structure_list & approved_acronyms

    # If a cell has nodes in RSPagl and RSPv, a coronal slice may create holes/gaps if RSPd is not in the structure list
    if any(['RSPagl' in s for s in cells_valid_structures]) and any(['RSPv' in s for s in cells_valid_structures]):
        cells_valid_structures.add("RSP")

    # If a structure is an iso-cortex leaf structure, we will add the parent structure so that all layers are
    # represented from each structure in cells_valid_structures
    final_structures = set()
    for struct_acr in cells_valid_structures:

        number_descendents = len(tree.descendants([id_by_acronym[struct_acr]])[0])
        # is a leaf node
        if number_descendents == 1:
            # add the parent structure
            parent_id = tree.parent_ids([id_by_acronym[struct_acr]])[0]
            parent_acronym = acronym_by_id[parent_id]
            final_structures.add(parent_acronym)
        else:
            final_structures.add(struct_acr)

    return list(final_structures)
