import logging
import vtk
import numpy as np
from ccf_streamlines.angle import find_closest_streamline
from ccf_streamlines.coordinates import coordinates_to_voxels
from neuron_morphology.transforms.affine_transform import (
    rotation_from_angle, affine_from_transform_translation, affine_from_translation, AffineTransform
)
from neuron_morphology.swc_io import Morphology
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from skimage.measure import find_contours
from scipy.interpolate import interpn
from scipy.spatial.distance import euclidean
from scipy.spatial.transform import Rotation
from vtk.util.numpy_support import vtk_to_numpy
from scipy import ndimage
from scipy.spatial import KDTree
from copy import copy
from collections import deque


def correct_upright_morph_orientation(upright_morphology, atlas_slice_orient_data):
    """
    Given atlas slice information, this will rotate the morphology so that it will subsequently 
    be quantified in a near coronal orientation. Where atlas slice information is data from either
    skeleton_keys.full_morph.angled_atlas_slice_for_morph or 
    skeleton_keys.full_morph.min_curvature_atlas_slice_for_morph

    :param upright_morphology: an upright (or layer aligned) neuron_morphology.Morphology
    :param atlas_slice_orient_data: dict with the following keys:
        "base_orientation": str or None - 'coronal' or 'parasagittal',
        "base_orientation_slice_angle": float or None, radians degree of the slice when base_orientation is not None
        "atlas_slice_off_coronal_angle": float or None, angle (degrees) needed to rotate the morphology to achieve coronal perspective

    :return: neuron_morphology.Morphology

    """
    morph = upright_morphology.clone()
    
    base_orientation = atlas_slice_orient_data['base_orientation']
    if base_orientation is not None:
        if base_orientation == 'coronal':
            # no rotation needed
            return morph
        
        elif base_orientation=="parasagittal":
            # because features in x are able to tolerate left/right handedness it does 
            # not matter if we rotate 90 or 270 degrees about the y axis
            n_degrees = 90
        else:
            msg = f"""
            unexpected value for base_orientation found in atlas_slice_orient_data ({base_orientation}). Expected
            values are ['parasagittal', 'coronal', None]
            """
            raise ValueError(msg)
                
    else:
        n_degrees = atlas_slice_orient_data['atlas_slice_off_coronal_angle']

    n_radians = np.radians(n_degrees)        
    rotation_matrix = rotation_from_angle(n_radians, axis=1)
    morph, _ = _rotate_morphology_around_soma(morph, rotation_matrix, None)
    
    return morph

def bfs_tree(st_node, morph):
    """
    breadth first traversal of tree, returns nodes in segment

    :param st_node: node to begin BFS traversal from.
    :param morph: neuron_morphology Morphology object
    :return: list of nodes in segment (including start node)
    """
    max_iterations = len(morph.nodes())
    queue = deque([st_node])
    nodes_in_segment = []
    seg_len = 0
    while len(queue) > 0:
        seg_len += 1
        if seg_len > max_iterations:
            return [], 0
        current_node = queue.popleft()
        nodes_in_segment.append(current_node)
        for ch_no in morph.get_children(current_node):
            queue.append(ch_no)

    return nodes_in_segment

def mean_spacing(coords):
    """
    Calculate the average inter-coordinate spacing of a streamline.

    Parameters:
        coords (numpy.ndarray): Input coordinates.

    Returns:
        float: Average inter-coordinate spacing.
    """
    dists = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
    return np.mean(dists)

def upsample_streamline_betwn(streamline, desired_spacing=0.5, max_iter=100):
    """
    Upsample a streamline until the desired spacing between coordinates is reached or maximum iterations are reached.

    Parameters:
        streamline (numpy.ndarray): Streamline coordinates.
        desired_spacing (float): Desired spacing between coordinates of the streamline.
        max_iter (int): Maximum number of iterations before stopping.

    Returns:
        numpy.ndarray: Upsampled streamline.
    """
    spacing = mean_spacing(streamline)
    it_ct = 0

    while desired_spacing < spacing:
        new_streamline = np.zeros((2 * len(streamline) - 1, 3))
        new_streamline[::2] = streamline
        new_streamline[1::2] = (streamline[:-1] + streamline[1:]) / 2
        streamline = new_streamline
        spacing = mean_spacing(streamline)
        it_ct += 1

        if it_ct > max_iter:
            break

    return streamline

def local_crop_cortical_morphology(morphology, 
                                   streamline_locate_coord, 
                                   closest_surface_voxel_file, 
                                   surface_paths_file,
                                   threshold=500):

    """Find the streamline nearest to `streamline_locate_coord`, upsample it and find nodes 
    from `morphology` that are within `threshold` distance of the streamline. 
    
    Parameters:
        morphology (neuron_morphology.morphology): Input morphology
        streamline_locate_coord (numpy.ndarray): reference coordinate to find streamling (typically the soma coordinate)
        closest_surface_voxel_file (str): path to closest_surface_voxel_file
        surface_paths_file (str): path to surface_paths_file
        threshold (float): allowed distance from streamline

    Returns:
        new_morph: (neuron_morphology.morphology): "local crop" of morphology
    """
    closest_streamline = find_closest_streamline(streamline_locate_coord,
                                            closest_surface_voxel_file,
                                            surface_paths_file
                                            )
    upsampled_streamline = upsample_streamline_betwn(closest_streamline)

    streamline_tree = KDTree(upsampled_streamline)
    morph_nodes = np.array([[n['x'],n['y'],n['z']] for n in morphology.nodes()])
    morph_ids = np.array([n['id'] for n in morphology.nodes()])

    dists,inds = streamline_tree.query(morph_nodes)
    
    keeping_inds = dists<threshold
    close_enough_node_ids = set(morph_ids[keeping_inds])
    too_far_node_ids = set(morph_ids[~keeping_inds])
    close_nodes = [node for ct, node in enumerate(morphology.nodes()) if keeping_inds[ct]]
    
    new_morph = Morphology(close_nodes, 
                           parent_id_cb=lambda x: x['parent'], 
                           node_id_cb=lambda x: x['id'])

    orphans = [n for n in new_morph.nodes() if n['parent'] in too_far_node_ids]
    disconnected_node_ids = []
    for orphan_no in orphans:
        downstream_nodes = bfs_tree(orphan_no, new_morph)
        for no in downstream_nodes:
            disconnected_node_ids.append(no['id'])

    keeping_nodes = [n for n in new_morph.nodes() if n['id'] not in disconnected_node_ids]
    new_morph = Morphology(keeping_nodes,
                            parent_id_cb=lambda x: x['parent'],
                            node_id_cb=lambda x: x['id'])

    return new_morph

def find_structures_morphology_occupies(rot_morph, atlas_slice, tree,
                                        structures_of_interest=["Isocortex"],
                                        resolution=10):
    """
    Given an atlas slice and morphology aligned to that atlas slice, this will remove structures from that
    atlas slice that the morphology does not occupy. The last step ensures that there are not disconnected
    structures arising from the curvature of the brain and the morphological masking implemented here. 
    
    :param rot_morph: morphology
        morphology that is in micron space that has been rotated to align with the atlas_slice
    :param atlas_slice: 2D array
        Region-annotated CCF atlas slice
    :param tree: structure tree
        from allensdk package
    :param structures_of_interest: list
        list of structures used to mask cell. E.g. if cell has node(s) in Pons but structures_of_interest
        list does not contain Pons or any Pons ontological ancestors, Pons will not be returned in
        final_structures

    :return: list
        list of structures the cell occupies that are descendants of structures_of_interest
    """


    id_by_acronym = tree.get_id_acronym_map()
    acronym_by_id = {v: k for k, v in id_by_acronym.items()}
    id_by_name = {v: k for k, v in tree.get_name_map().items()}

    micron_to_voxel = AffineTransform.from_list([0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1, 0, 0, 0])
    voxel_rot_morph = micron_to_voxel.transform_morphology(rot_morph.clone())
    voxel_rot_morph_nodes = np.array([[n['x'], n['y'], n['z']] for n in voxel_rot_morph.nodes()]).astype(int)

    structure_ids_morph_occupies = set(atlas_slice[voxel_rot_morph_nodes[:, 0], voxel_rot_morph_nodes[:, 1]])
    structure_acronyms_morph_occupies = set([acronym_by_id[i] for i in structure_ids_morph_occupies if i != 0])

    structure_ids_in_atlas_slice = set(atlas_slice.flatten()) 

    # find the set of structures that are approved (i.e. are descendants of the structures in structures_of_interest list)
    approved_acronyms = set()
    approved_ids = set()
    for roi_struct_name in structures_of_interest:
        roi_id = id_by_name[roi_struct_name]
        all_roi_descendants = tree.descendants([roi_id])[0]
        for struct_dict in all_roi_descendants:
            approved_acronyms.add(struct_dict['acronym'])
            approved_ids.add(struct_dict['id'])

    valid_ids_in_atlas_slice_before_filter = approved_ids & structure_ids_in_atlas_slice
    valid_ids_morph_occupies = structure_ids_morph_occupies & approved_ids

    # If a morphology only occupies Layers 1 through 4, we still need layer drawings for the deeper layers. 
    # This will add ontological sibling structures for each leaf node
    to_add = set()
    for struct_id in valid_ids_morph_occupies:
        parent_id = tree.parent_ids([struct_id])[0]
        sibling_ids = tree.descendant_ids([parent_id])[0][1:]
        for s_id in sibling_ids:
            to_add.add(s_id)
            
    valid_ids_morph_occupies = valid_ids_morph_occupies | to_add
    valid_acronyms_morph_occupies = [acronym_by_id[i] for i in valid_ids_morph_occupies]

    # Generate slice that has only valid (Isocortex) structures
    valid_atlas_slice = copy(atlas_slice)
    valid_atlas_slice[np.isin(valid_atlas_slice, list(valid_ids_in_atlas_slice_before_filter), invert=True)] = 0

    # Generate slice that has only valid structures morphology occupies
    filtered_atlas_slice = copy(atlas_slice)
    filtered_atlas_slice[np.isin(filtered_atlas_slice, list(valid_ids_morph_occupies), invert=True)] = 0


    # Run connected components on filtered_atlas_slice to only keep groups of structures that the morphology 
    # occupies. The intention here is to remove "islands" that form in the atlas slice when we 
    # filter out structures the morphology does not occupy. This can happen with unlucky curvature of the brain
    # and the angle of the slice selected. For example in the 2024 Sorensen et al manuscript, cell 
    # 182725_7649-X3440-Y17884_reg.swc exists in VISp with some local axon in dorsal and ventral parts of RSP.
    # Given the atlas slice extracted that minimizes the curvature between pia and wm, RSP structures are not 
    # one continuous connected components in the atlas slice. They appear in separate island. Since our morphology
    # has axon in RSP, we need to remove the island of RSP that the cell does not innervate, otherwise we will
    # have a Multipolygon shapely error. 
    labels, num_components = ndimage.label(filtered_atlas_slice)
    morph_nodes = np.array([[n['x'], n['y'], n['z']] for n in rot_morph.nodes()])
    morph_nodes_voxel = coordinates_to_voxels(morph_nodes, resolution = (resolution,resolution,resolution))

    morph_mask = np.zeros_like(filtered_atlas_slice)
    morph_mask[morph_nodes_voxel[:,0], morph_nodes_voxel[:,1]] = 1

    # find the connected components that the morphology occupies
    morph_masked_labels = np.multiply(labels, morph_mask)
    keeping_conn_components = [i for i in set(morph_masked_labels.flatten()) if i != 0]

    mask = np.isin(labels, keeping_conn_components)

    neuron_masked_filtered_atlas_slice = np.multiply(mask, filtered_atlas_slice)
    
    return neuron_masked_filtered_atlas_slice


def check_coord_out_of_cortex(coordinate, structure_id, atlas_volume, closest_surface_voxel_file, surface_paths_file,
                              tree):
    """
    Check if a given ccf coordinate (microns) is located outside a certain CCF structure

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
    Given an out of cortex voxel, this will find the nearest iso-cortex voxel. This will take the 3-d isocortex annotation and
    erode it one step leaving us the "shell" of isocortex. The nearest point

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
    additional_data : nx3 array, None
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
    morphology, additional_data = _rotate_morphology_around_soma(morphology, rotation_tilt_matrix, additional_data)
    morphology, additional_data = _swap_morphology_coordinates(morphology, coordinate_swap_matrix, additional_data)

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
    additional_data : nx3 array, None
        Additional coordinates to apply the same rotation to. Specifically useful when
        a cells soma is in the white matter and the nearest iso-cortex voxel needs to be
        rotated to line up with the angled atlas image.

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
    morphology, additional_data = _swap_morphology_coordinates(morphology, coordinate_swap_matrix, additional_data)

    return morphology, additional_data


def align_morphology_to_drawings(morphology, atlas_slice, closest_cortex_node=None, atlas_resolution=10.):
    """ Move the soma to the center of a set of drawings

    Parameters
    ----------
    morphology : Morphology
        Full brain morphology, aligned to CCF (micron scale)
    atlas_slice : 2D array
        Angled atlas slice with CCF region IDs as values
    atlas_resolution : float, default 10
        Voxel size of atlas volume (microns)
    closest_cortex_node : array, default None
        If the morphologys soma is outside the cortex, this value is used as a proxy
        for the soma location. It represents the closest isocortex node to the cells soma.

    Returns
    -------
    aligned_morphology : Morphology
        Translated morphology
    """
    if closest_cortex_node is None:
        morph_soma = morphology.get_soma()
        soma_coords = np.array([morph_soma['x'], morph_soma['y'], morph_soma['z']])
    else:
        soma_coords = closest_cortex_node

    # Align morphology to center of drawings
    translation = np.array([
        -soma_coords[0] + atlas_resolution * atlas_slice.shape[0] / 2,
        -soma_coords[1] + atlas_resolution * atlas_slice.shape[1] / 2,
        -soma_coords[2]
    ])
    translation_affine = affine_from_translation(translation)
    T_translate = AffineTransform(translation_affine)
    T_translate.transform_morphology(morphology)
    if closest_cortex_node is not None:
        closest_cortex_node = T_translate.transform(closest_cortex_node)

    return morphology, closest_cortex_node

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

def _swap_morphology_coordinates(morphology, M, additional_data=None):
    """Use matrix to switch coordinates"""

    # Swap coordinates
    swap_affine = affine_from_transform_translation(transform=M)
    T_swap = AffineTransform(swap_affine)
    T_swap.transform_morphology(morphology)
    if additional_data is not None:
        additional_data = T_swap.transform(additional_data)

    return morphology, additional_data

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
    closest_cortex_node : array, default None
        If the morphology's soma is outside the cortex, this value is required to
        find the nearest streamline. It represents the closest isocortex node to
        the cells soma.
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
            logging.info(
                f"Auto selecting parasagittal orientation ({para_angle_deg:.2f} angle vs {coronal_angle_deg:.2f} for coronal)")
        else:
            norm_vec = coronal_norm_vec
            base_orientation = 'coronal'
            logging.info(
                f"Auto selecting coronal orientation ({coronal_angle_deg:.2f} angle vs {para_angle_deg:.2f} for parasagittal)")
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
                                        closest_cortex_node=None, atlas_resolution=10., ):
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
    closest_cortex_node : array, default None
        If the morphology's soma is outside the cortex, this value is used as a proxy
        for the soma location. It represents the closest isocortex node to the cells soma.
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

    # To ensure that cells appear in a near coronal perspective, we need 
    # to find the angle between the vector normal to the 3d slice and the
    # anterior-posterior axis so that we can rotate the upright and layer
    # aligned file post-hoc. 
    
    # Find the corners that define the original unrotated slice grid
    x_min, x_max = np.min(slice_grid[0]), np.max(slice_grid[0])
    y_min, y_max = np.min(slice_grid[1]), np.max(slice_grid[1])
    z_min, z_max = np.min(slice_grid[2]), np.max(slice_grid[2])

    corner1 = [x_min, y_min, z_min]
    corner2 = [x_min, y_max, z_min]
    corner3 = [x_max, y_min, z_min]
    corner4 = [x_max, y_max, z_min]
    corners = np.array([corner1,corner2,corner3,corner4])
    corner_coords = np.array([
    corners[:,0].flatten(),
    corners[:,1].flatten(),
    corners[:,2].flatten()
    ])
    
    # Rotate the corners so we can define the rotated plane
    rot_corners_coords = (M @ (corner_coords - soma_coords_for_atlas[:, np.newaxis])).T + soma_coords_for_atlas
    
    # Compute vectors from the corners
    v1 = rot_corners_coords[1] - rot_corners_coords[0]
    v2 = rot_corners_coords[2] - rot_corners_coords[0]

    # Compute the normal vector to the plane
    normal_vector = np.cross(v1, v2)
    normal_vector /= np.linalg.norm(normal_vector)  

    # Project the normal vector onto the x-z plane 
    normal_vector_2d = np.array([normal_vector[0], normal_vector[2]])

    # if the vector is facing away from the brain in the A-P axis,
    # flip so that it points towards the brain 
    if normal_vector[0]<0:
        normal_vector_2d = -1*normal_vector_2d

    # tells us which direction to rotate the morph post-hoc
    sign_flip = 1 if normal_vector_2d[1]<0 else -1

    # Define the anterior-posterior axis vector
    vector_xz = np.array([1,0])

    # Compute the dot product between the two vectors
    dot_product = np.dot(normal_vector_2d, vector_xz)

    # Compute the angle between the vectors
    angle_radians = np.arccos(dot_product)
    angle_degrees = np.degrees(angle_radians)
    angle_degrees = sign_flip*angle_degrees
    
    return atlas_slice, q, angle_degrees

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
            # Find the contour with largest perimeter
            max_perim = 0
            for c in contours:
                if c.shape[0] < min_contour_pts:
                    # Skip very small contours
                    continue
                perimeter = len(c)
                if max_perim < perimeter:
                    boundaries[name] = c
                    max_perim = perimeter
    return boundaries
