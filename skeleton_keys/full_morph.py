import logging
import numpy as np
from ccf_streamlines.angle import find_closest_streamline
from neuron_morphology.transforms.affine_transform import (
    rotation_from_angle, affine_from_transform_translation, affine_from_translation, AffineTransform
)
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from skimage.measure import find_contours
from scipy.interpolate import interpn
from scipy.spatial.distance import euclidean


def rotate_morphology_for_drawings(morphology, angle_for_atlas_slice, base_orientation='coronal'):
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

    # center on soma before tilt
    soma_morph = morphology.get_soma()
    translation_to_origin = np.array([-soma_morph['x'], -soma_morph["y"], -soma_morph["z"]])
    translation_back_to_soma_location = -translation_to_origin

    translation_affine = affine_from_translation(translation_to_origin)
    T_translate = AffineTransform(translation_affine)
    T_translate.transform_morphology(morphology)

    # Rotate
    rotation_tilt_matrix = rotation_from_angle(-angle_for_atlas_slice, axis=rot_axis)
    rotation_tilt_affine = affine_from_transform_translation(transform=rotation_tilt_matrix)
    T_rotate = AffineTransform(rotation_tilt_affine)
    T_rotate.transform_morphology(morphology)

    # Move back
    translation_affine = affine_from_translation(translation_back_to_soma_location)
    T_translate = AffineTransform(translation_affine)
    T_translate.transform_morphology(morphology)

    # Swap coordinates
    swap_affine = affine_from_transform_translation(transform=coordinate_swap_matrix)
    T_swap = AffineTransform(swap_affine)
    T_swap.transform_morphology(morphology)

    return morphology


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
        base_orientation='auto',
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

    Returns
    -------
    atlas_slice : 2D array
        Angled atlas slice with CCF region IDs as values
    angle_rad : float
        Angle of slice from original plane (radians)
    slice_orientation : str
        Orientation of slice (for auto selection; otherwise matches base_orientation)
    """

    morph_soma = morph.get_soma()
    soma_coords = np.array([morph_soma['x'], morph_soma['y'], morph_soma['z']])

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

    uniq_structs, inv = np.unique(atlas_image, return_inverse=True)

    structure_layer_mapping = {}
    for struct_set_id in layer_structure_sets.values():
        structures_of_set = [d['id'] for d in tree.get_structures_by_set_id([struct_set_id])]
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


def find_layer_outlines(atlas_image):
    """ Identify boundaries of each layer from a 2D atlas image

    Atlas image must contain the structure set IDs of the cortical layers

    Parameters
    ----------
    atlas_image : 2D array
        Atlas image

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
            # No contours found
            boundaries[name] = np.array([])
        elif len(contours) == 1:
            boundaries[name] = contours[0]
        else:
            # Find the biggest contour
            max_len = 0
            for c in contours:
                if len(c) > max_len:
                    boundaries[name] = c
                    max_len = len(c)
    return boundaries
