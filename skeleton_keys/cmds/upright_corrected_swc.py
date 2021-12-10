import json
import numpy as np
import pandas as pd
import argschema as ags
from neuron_morphology.swc_io import morphology_from_swc, morphology_to_swc
from neuron_morphology.transforms.pia_wm_streamlines.calculate_pia_wm_streamlines import (
    run_streamlines,
)
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


class UprightCorrectedSwcSchema(ags.ArgSchema):
    specimen_id = ags.fields.Integer(description="Specimen ID")
    swc_path = ags.fields.InputFile(
        description="path to SWC file (optional)", default=None, allow_none=True
    )
    output_file = ags.fields.OutputFile(default="output.swc")
    correct_for_shrinkage = ags.fields.Boolean(
        default=True,
        description="Whether to correct for shrinkage",
    )
    correct_for_slice_angle = ags.fields.Boolean(
        default=True,
        description="Whether to correct for slice angle",
    )
    surface_and_layers_file = ags.fields.InputFile(
        description="JSON file with surface and layer polygon paths",
        default=None,
        allow_none=True,
    )
    closest_surface_voxel_file = ags.fields.InputFile(
        default=None,
        allow_none=True,
        description="Closest surface voxel reference HDF5 file for slice angle calculation",
    )
    surface_paths_file = ags.fields.InputFile(
        default=None,
        allow_none=True,
        description="Surface paths (streamlines) HDF5 file for slice angle calculation",
    )


def main(args):
    # Get the path to the SWC file
    specimen_id = args["specimen_id"]
    swc_path = args["swc_path"]
    if swc_path is None:
        swc_path = swc_paths_from_database([specimen_id])[specimen_id]

    # Load the reference layer depths
    with open(args["layer_depths_file"], "r") as f:
        avg_layer_depths = json.load(f)

    layer_list = args["layer_list"]

    # Get pia, white matter, soma, and layers
    surface_and_layers_file = args["surface_and_layers_file"]
    if surface_and_layers_file is not None:
        with open(surface_and_layers_file, "r") as f:
            surfaces_and_paths = json.load(f)
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
    else :
        # just upright the morphology
        rotation_upright_matrix = rotation_from_angle(upright_angle, axis=2)
        rotation_upright_affine = affine_from_transform_translation(
            transform=rotation_upright_matrix
        )
        T_rotate = AffineTransform(rotation_upright_affine)
        morph = T_rotate.transform_morphology(morph)

    # save to swc
    output_file = args["output_file"]
    morphology_to_swc(morph, output_file)


def console_script():
    module = ags.ArgSchemaParser(schema_type=UprightCorrectedSwcSchema)
    main(module.args)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=UprightCorrectedSwcSchema)
    main(module.args)
