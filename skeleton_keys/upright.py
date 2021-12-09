import numpy as np
from neuron_morphology.transforms.affine_transform import (
    rotation_from_angle, affine_from_transform_translation, affine_from_translation, AffineTransform)


def upright_corrected_morph(
    morphology,
    upright_angle,
    slice_angle,
    flip_status,
    shrink_factor):
    """ Rotate morphology upright and correct for shrinkage and slice angle

    Parameters
    ----------
    morphology : Morphology
        Original morphology (micron scale, in image alignment)
    upright_angle : float
        Angle to rotate to upright the morphology (radians)
    slice_angle : float
        Angle between actual slice and plane parallel to streamlines (radians)
    flip_status : integer
        1 = slice was flipped before recording; -1 = slice was not flipped
    shrink_factor : float
        Factor to multiple z-values by to correct for slice shrinkage

    Returns
    -------
    corrected_morph : Morphology
        Morphology in upright orientation after correcting for shrinkage
        and slice angle
    """
    tilt = -slice_angle * flip_status

    # center on soma
    soma_morph = morphology.get_soma()
    translation_to_origin = np.array([-soma_morph['x'], -soma_morph["y"], 0])
    translation_affine = affine_from_translation(translation_to_origin)
    T_translate = AffineTransform(translation_affine)
    T_translate.transform_morphology(morphology)

    # Put the cell upright
    rotation_upright_matrix = rotation_from_angle(upright_angle, axis=2)
    rotation_upright_affine = affine_from_transform_translation(transform=rotation_upright_matrix)
    T_rotate = AffineTransform(rotation_upright_affine)
    T_rotate.transform_morphology(morphology)

    # correct for z-shrinkage
    aff_mat = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., shrink_factor]])
    scale_affine = affine_from_transform_translation(transform=aff_mat)
    T_expand = AffineTransform(scale_affine)
    T_expand.transform_morphology(morphology)

    # center on z before tilt
    soma_morph = morphology.get_soma()
    new_soma_z = soma_morph["z"]
    translation_in_z = np.array([0, 0, -new_soma_z])
    translation_affine = affine_from_translation(translation_in_z)
    T_translate = AffineTransform(translation_affine)
    T_translate.transform_morphology(morphology)

    # correct for slice angle tilt
    rotation_tilt_matrix = rotation_from_angle(tilt, axis=0)
    rotation_tilt_affine = affine_from_transform_translation(transform=rotation_tilt_matrix)
    T_rotate = AffineTransform(rotation_tilt_affine)
    T_rotate.transform_morphology(morphology)

    # move back to z before tilt
    translation_in_z = np.array([0, 0, new_soma_z])
    translation_affine = affine_from_translation(translation_in_z)
    T_translate = AffineTransform(translation_affine)
    T_translate.transform_morphology(morphology)

    return morphology


def corrected_without_uprighting_morph(
    morphology,
    upright_angle,
    slice_angle,
    flip_status,
    shrink_factor):
    """ Correct for shrinkage and slice angle without uprighting

    Note, though, that upright_angle must be provided so that the slice angle
    corrections are correctly applied

    Parameters
    ----------
    morphology : Morphology
        Original morphology (micron scale, in image alignment)
    upright_angle : float
        Angle to rotate to upright the morphology (radians)
    slice_angle : float
        Angle between actual slice and plane parallel to streamlines (radians)
    flip_status : integer
        1 = slice was flipped before recording; -1 = slice was not flipped
    shrink_factor : float
        Factor to multiple z-values by to correct for slice shrinkage

    Returns
    -------
    corrected_morph : Morphology
        Morphology in upright orientation after correcting for shrinkage
        and slice angle
    """
    morphology = upright_corrected_morph(
        morphology,
        upright_angle,
        slice_angle,
        flip_status,
        shrink_factor)

    # undo upright transform
    rotation_upright_matrix = rotation_from_angle(-upright_angle, axis=2)
    rotation_upright_affine = affine_from_transform_translation(transform=rotation_upright_matrix)
    T_rotate = AffineTransform(rotation_upright_affine)
    T_rotate.transform_morphology(morphology)

    return morphology

