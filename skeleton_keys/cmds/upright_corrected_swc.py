import json
import shapely
import numpy as np
import pandas as pd
import argschema as ags
from neuron_morphology.swc_io import morphology_from_swc, morphology_to_swc
from neuron_morphology.transforms.pia_wm_streamlines.calculate_pia_wm_streamlines import (
    run_streamlines,
)
from neuron_morphology.snap_polygons.geometries import Geometries
from neuron_morphology.transforms.upright_angle.compute_angle import get_upright_angle
from neuron_morphology.transforms.affine_transform import (
    rotation_from_angle,
    affine_from_transform_translation,
    affine_from_translation,
    AffineTransform,
)
from skeleton_keys.database_queries import (
    query_for_image_series_id,
    swc_paths_from_database,
    pia_wm_soma_from_database,
    shrinkage_factor_from_database,
    query_pinning_info,
    determine_flip_switch,
)
from skeleton_keys.slice_angle import slice_angle_tilt
from skeleton_keys.upright import upright_corrected_morph
from skeleton_keys.drawings import (
    convert_and_translate_snapped_to_microns,
    remove_duplicate_coordinates_from_drawings,
)
from skeleton_keys.layer_alignment import (
    setup_interpolator_without_nan,
    path_dist_from_node,
)
from skeleton_keys.io import read_json_file, load_default_layer_template
from skeleton_keys import cloudfields


class UprightCorrectedSwcSchema(ags.ArgSchema):
    specimen_id = ags.fields.Integer(description="Specimen ID")
    swc_path = cloudfields.InputFile(
        description="path to SWC file (optional)", default=None, allow_none=True
    )
    output_file = cloudfields.OutputFile(default="output.swc")
    correct_for_shrinkage = ags.fields.Boolean(
        default=True,
        description="Whether to correct for shrinkage",
    )
    correct_for_slice_angle = ags.fields.Boolean(
        default=True,
        description="Whether to correct for slice angle",
    )
    surface_and_layers_file = cloudfields.InputFile(
        description="JSON file with surface and layer polygon paths",
        default=None,
        allow_none=True,
    )
    closest_surface_voxel_file = cloudfields.InputFile(
        default=None,
        allow_none=True,
        description="Closest surface voxel reference HDF5 file for slice angle calculation",
    )
    surface_paths_file = cloudfields.InputFile(
        default=None,
        allow_none=True,
        description="Surface paths (streamlines) HDF5 file for slice angle calculation",
    )


def soma_distance_from_pia(pia_surface, depth_field, gradient_field, translation):
    """Calculate the distance (in microns) from the soma to the pia"""
    # Create pia surface geometry
    surfs = Geometries()
    surfs.register_surface("pia", pia_surface["path"])
    surfs = surfs.to_json()

    # convert to micron scale and translate to soma-centered depth field
    surfs = convert_and_translate_snapped_to_microns(
        surfs, pia_surface["resolution"], translation
    )
    surfs_dict = {
        s["name"]: shapely.geometry.LineString(s["path"]) for s in surfs["surfaces"]
    }

    # Set up interpolators for navigating fields
    depth_interp = setup_interpolator_without_nan(
        depth_field, None, method="linear", bounds_error=False, fill_value=None
    )
    dx_interp = setup_interpolator_without_nan(
        gradient_field, "dx", method="linear", bounds_error=False, fill_value=None
    )
    dy_interp = setup_interpolator_without_nan(
        gradient_field, "dy", method="linear", bounds_error=False, fill_value=None
    )

    # Field is soma centered, so find the distance to pia from (0, 0)
    soma_dist_to_pia = path_dist_from_node(
        (0, 0),
        depth_interp,
        dx_interp,
        dy_interp,
        surfs_dict["pia"],
        step_size=1.0,
        max_iter=1000,
    )
    return soma_dist_to_pia


def main(args):
    # Get the path to the SWC file
    specimen_id = args["specimen_id"]
    swc_path = args["swc_path"]
    if swc_path is None:
        swc_path = swc_paths_from_database([specimen_id])[specimen_id]

    # Get pia, white matter, soma, and layers
    surface_and_layers_file = args["surface_and_layers_file"]
    if surface_and_layers_file is not None:
        surfaces_and_paths = read_json_file(surface_and_layers_file)
        pia_surface = surfaces_and_paths["pia_path"]
        wm_surface = surfaces_and_paths["wm_path"]
        soma_drawing = surfaces_and_paths["soma_path"]
    else:
        # Query for image series ID
        imser_id = query_for_image_series_id(specimen_id)

        # Query for pia, white matter, and soma
        pia_surface, wm_surface, soma_drawing = pia_wm_soma_from_database(
            specimen_id, imser_id
        )

    # Remove any duplicate coordinates from surfaces
    pia_surface = remove_duplicate_coordinates_from_drawings(pia_surface)
    wm_surface = remove_duplicate_coordinates_from_drawings(wm_surface)

    # Load the morphology
    morph = morphology_from_swc(swc_path)

    # Determine the streamline field and upright angle

    # Construct strings from paths
    pia_path = ",".join(["{},{}".format(x, y) for x, y in pia_surface["path"]])
    wm_path = ",".join(["{},{}".format(x, y) for x, y in wm_surface["path"]])
    soma_path = ",".join(["{},{}".format(x, y) for x, y in soma_drawing["path"]])
    resolution = pia_surface["resolution"]

    depth_field, gradient_field, translation = run_streamlines(
        pia_path,
        wm_path,
        resolution,
        soma_path,
    )
    upright_angle = get_upright_angle(gradient_field)

    # Correct for shrinkage and/or slice angle if requested
    if args["correct_for_shrinkage"] or args["correct_for_slice_angle"]:
        if args["correct_for_shrinkage"]:
            shrink_factor = shrinkage_factor_from_database(morph, specimen_id)
        else:
            shrink_factor = 1

        if args["correct_for_slice_angle"]:
            pin_df = pd.DataFrame.from_records(query_pinning_info())
            slice_angle = slice_angle_tilt(
                pin_df,
                specimen_id,
                args["closest_surface_voxel_file"],
                args["surface_paths_file"],
            )
            flip_status = determine_flip_switch(morph, specimen_id)

        else:
            slice_angle = 0
            flip_status = 1

        morph = upright_corrected_morph(
            morph, upright_angle, slice_angle, flip_status, shrink_factor
        )
    else:
        # just upright the morphology
        rotation_upright_matrix = rotation_from_angle(upright_angle, axis=2)
        rotation_upright_affine = affine_from_transform_translation(
            transform=rotation_upright_matrix
        )
        T_rotate = AffineTransform(rotation_upright_affine)
        morph = T_rotate.transform_morphology(morph)

    # Set to correct depth from pia
    soma_dist_to_pia = soma_distance_from_pia(
        pia_surface, depth_field, gradient_field, translation
    )

    translation_for_soma = np.array([0, -soma_dist_to_pia, 0])
    translation_affine = affine_from_translation(translation_for_soma)
    T_translate = AffineTransform(translation_affine)
    T_translate.transform_morphology(morph)

    # save to swc
    output_file = args["output_file"]
    morphology_to_swc(morph, output_file)


def console_script():
    module = ags.ArgSchemaParser(schema_type=UprightCorrectedSwcSchema)
    main(module.args)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=UprightCorrectedSwcSchema)
    main(module.args)
