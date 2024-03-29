import json
import logging
import numpy as np
import pandas as pd
import argschema as ags
from skeleton_keys.layer_alignment import (
    layer_aligned_y_values_for_morph,
    cortex_thickness_aligned_y_values_for_morph,
)
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
from neuron_morphology.morphology import Morphology
from skeleton_keys.database_queries import (
    query_for_image_series_id,
    swc_paths_from_database,
    pia_wm_soma_from_database,
    layer_polygons_from_database,
    shrinkage_factor_from_database,
    query_pinning_info,
    determine_flip_switch,
)
from skeleton_keys.slice_angle import slice_angle_tilt
from skeleton_keys.drawings import (
    snap_hand_drawn_polygons,
    convert_and_translate_snapped_to_microns,
    remove_duplicate_coordinates_from_drawings,
)
from skeleton_keys.upright import corrected_without_uprighting_morph
from skeleton_keys.io import read_json_file,load_default_layer_template
from skeleton_keys import cloudfields
from skeleton_keys.full_morph import correct_upright_morph_orientation


class LayerAlignedSwcSchema(ags.ArgSchema):
    specimen_id = ags.fields.String(description="Specimen ID")
    swc_path = cloudfields.InputFile(
        description="path to SWC file (optional)", default=None, allow_none=True
    )
    layer_depths_file = cloudfields.InputFile(default=None, allow_none=True)
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
    layer_list = ags.fields.List(
        ags.fields.String,
        description="List of layer names in order",
        default=["Layer1", "Layer2/3", "Layer4", "Layer5", "Layer6a", "Layer6b"],
        cli_as_single_argument=True,
    )
    align_morph_to_layer_drawings = ags.fields.Bool(
        default=True,
        description=(
            "If the morphology and layer drawings are not alginmened (not overlaping), this flag  "
            "should be set True to translate the morphology to the mean 'soma_drawing' value "
            "found in layer drawings. I.e. if you plot layer polygons and the morphology and they are "
            "not aligned, this should be set True. This is standard in LIMS/patch-seq data. However, in "
            "ccf-registered data extracted from skeleton-keys, this is not True, so this should be set False."
        )
    )
    ccf_slice_orientation_correction = ags.fields.Boolean(
        default=True,
        description=(
            "When skeleton_keys.cmds.full_morph_layer_drawings was used to generate --swc_path and "
            "--surface_and_layers_file, this will rotate the upright morphology about the "
            "soma-centered y-axis so that the result is near coronal orientation."
        )
    )


def main(args):

    # Get the path to the SWC file
    specimen_id = args["specimen_id"]
    swc_path = args["swc_path"]
    if swc_path is None:
        specimen_id = int(specimen_id)
        swc_path = swc_paths_from_database([specimen_id])[specimen_id]

    # Load the reference layer depths
    layer_depths_file = args["layer_depths_file"]
    if layer_depths_file:
       avg_layer_depths = read_json_file(args["layer_depths_file"])
    else:
        avg_layer_depths = load_default_layer_template()

    layer_list = args["layer_list"]

    # Get pia, white matter, soma, and layers
    surface_and_layers_file = args["surface_and_layers_file"]
    atlas_slice_orient_data = None
    if surface_and_layers_file is not None:
        surfaces_and_paths = read_json_file(surface_and_layers_file)
        pia_surface = surfaces_and_paths["pia_path"]
        wm_surface = surfaces_and_paths["wm_path"]
        soma_drawing = surfaces_and_paths["soma_path"]
        layer_polygons = surfaces_and_paths["layer_polygons"]
        if "atlas_slice_orientation" in surfaces_and_paths:
            atlas_slice_orient_data = surfaces_and_paths['atlas_slice_orientation']
    else:
        # Query for image series ID
        specimen_id = int(specimen_id)
        imser_id = query_for_image_series_id(specimen_id)

        # Query for pia, white matter, and soma
        pia_surface, wm_surface, soma_drawing = pia_wm_soma_from_database(
            specimen_id, imser_id
        )

        # Query for layers
        layer_polygons = layer_polygons_from_database(imser_id)

    # Remove any duplicate coordinates from surfaces
    pia_surface = remove_duplicate_coordinates_from_drawings(pia_surface)
    wm_surface = remove_duplicate_coordinates_from_drawings(wm_surface)

    # Check that layer polygons exist
    if len(layer_polygons) < 1:
        logging.warning(
            f"No layer drawings found for {specimen_id}; will instead align to cortex depth"
        )
        no_layers = True
    else:
        no_layers = False

    # Load the morphology
    morph = morphology_from_swc(swc_path)

    # Determine the streamline field and upright angle

    # Construct strings from paths
    pia_path = ",".join(["{},{}".format(x, y) for x, y in pia_surface["path"]])
    wm_path = ",".join(["{},{}".format(x, y) for x, y in wm_surface["path"]])
    soma_path = ",".join(["{},{}".format(x, y) for x, y in soma_drawing["path"]])
    resolution = pia_surface["resolution"]
    
    # align morphology to layer drawings, if necessary
    mean_soma = np.array(soma_drawing["path"]).mean(axis=0) * resolution
    if args['align_morph_to_layer_drawings']:
        curr_soma = morph.get_soma()
        to_drawing_aff = [mean_soma[0] - curr_soma['x'], mean_soma[1]-curr_soma['y'], 0]
        to_drawing_soma_affine = affine_from_translation(to_drawing_aff)
        AffineTransform(to_drawing_soma_affine).transform_morphology(morph) 
            
    logging.info(f"Calculating depth field for {specimen_id}")
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
            logging.info("Calculating shrinkage correction factor")
            shrink_factor = shrinkage_factor_from_database(morph, specimen_id)
        else:
            shrink_factor = 1

        if args["correct_for_slice_angle"]:
            logging.info("Determining slice angle")
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

        morph = corrected_without_uprighting_morph(
            morph, upright_angle, slice_angle, flip_status, shrink_factor
        )
        # because corrected_without_uprighting_morph moves the morphologys soma to the origin, 
        # the translation_to_origin below would no longer be valid without updating mean_soma here
        mean_soma = [morph.get_soma()['x'], morph.get_soma()['y']]

    # align morph to depth field
    # depth field is centered at origin
    translation_to_origin = np.array([-mean_soma[0], -mean_soma[1], 0])
    translation_affine = affine_from_translation(translation_to_origin)
    T_translate = AffineTransform(translation_affine)
    T_translate.transform_morphology(morph)
    # get aligned y-values for morph
    if no_layers:
        logging.info("Calculating cortex-thickness adjusted depths for all points")
        y_value_info = cortex_thickness_aligned_y_values_for_morph(
            morph, avg_layer_depths, depth_field
        )
    else:
        # snap together hand-drawn layer borders and determine pia/wm sides
        snapped_polys_surfs = snap_hand_drawn_polygons(
            layer_polygons, pia_surface, wm_surface, layer_list
        )

        # convert to micron scale and translate to soma-centered depth field
        snapped_polys_surfs = convert_and_translate_snapped_to_microns(
            snapped_polys_surfs, resolution, translation
        )

        logging.info("Calculating layer-aligned depths for all points")
        y_value_info = layer_aligned_y_values_for_morph(
            morph,
            avg_layer_depths,
            layer_list,
            depth_field,
            gradient_field,
            snapped_polys_surfs,
        )

    # upright the morphology
    rotation_upright_matrix = rotation_from_angle(upright_angle, axis=2)
    rotation_upright_affine = affine_from_transform_translation(
        transform=rotation_upright_matrix
    )
    T_rotate = AffineTransform(rotation_upright_affine)
    upright_morph = T_rotate.transform_morphology(morph)

    # assign new y-values
    node_list = []
    for node in upright_morph.nodes():
        if node["id"] in y_value_info:
            node["y"] = y_value_info[node["id"]]
            if node["parent"] not in y_value_info and node["parent"] != -1:
                node["parent"] = -1
            node_list.append(node)

    # build new morphology from reassigned nodes
    aligned_morph = Morphology(
        node_list,
        node_id_cb=lambda node: node["id"],
        parent_id_cb=lambda node: node["parent"],
    )

    # rotate full_morph generated cell so it is near coronal orientation
    if args['ccf_slice_orientation_correction'] and atlas_slice_orient_data is not None:
        aligned_morph = correct_upright_morph_orientation(aligned_morph, atlas_slice_orient_data)
      
    # save to swc
    output_file = args["output_file"]
    morphology_to_swc(aligned_morph, output_file)


def console_script():
    module = ags.ArgSchemaParser(schema_type=LayerAlignedSwcSchema)
    main(module.args)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=LayerAlignedSwcSchema)
    main(module.args)
