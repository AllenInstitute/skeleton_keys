import json
import logging
import numpy as np
import pandas as pd
import argschema as ags
from neuron_morphology.transforms.pia_wm_streamlines.calculate_pia_wm_streamlines import (
    run_streamlines, convert_path_str_to_list
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
)
from skeleton_keys.upright import corrected_without_uprighting_morph
from skeleton_keys.io import load_default_layer_template
from skeleton_keys.layer_alignment import layer_aligned_y_values, cortex_thickness_aligned_y_values


class LayerAlignedCoordsSchema(ags.ArgSchema):
    coordinate_file = ags.fields.InputFile(
        description="CSV file with three coordinate columns")
    layer_depths_file = ags.fields.InputFile(default=None, allow_none=True)
    output_file = ags.fields.OutputFile(
        description="CSV file with adjusted coordinates",
        default="output.csv")
    coordinate_column_prefix = ags.fields.String(
        description="common prefix for coordinate columns",
        default=""
    )
    coordinate_column_suffix = ags.fields.String(
        description="common suffix for coordinate columns",
        default=""
    )
    surface_and_layers_file = ags.fields.InputFile(
        description="JSON file with surface and layer polygon paths",
        default=None,
        allow_none=False,
    )
    layer_list = ags.fields.List(
        ags.fields.String,
        description="List of layer names in order",
        default=["Layer1", "Layer2/3", "Layer4", "Layer5", "Layer6a", "Layer6b"],
        cli_as_single_argument=True,
    )


def main(args):

    # Get the path to the SWC file
    coordinate_file = args["coordinate_file"]

    # Load the reference layer depths
    layer_depths_file = args['layer_depths_file']
    if layer_depths_file:
        with open(layer_depths_file, "r") as f:
            avg_layer_depths = json.load(f)
    else:
        avg_layer_depths = load_default_layer_template()


    layer_list = args["layer_list"]

    # Get pia, white matter, soma, and layers
    surface_and_layers_file = args["surface_and_layers_file"]
    with open(surface_and_layers_file, "r") as f:
        surfaces_and_paths = json.load(f)
    pia_surface = surfaces_and_paths["pia_path"]
    wm_surface = surfaces_and_paths["wm_path"]
    layer_polygons = surfaces_and_paths["layer_polygons"]
    if "soma_path" in surfaces_and_paths:
        soma_drawing = surfaces_and_paths["soma_path"]
    else:
        soma_drawing = None

    # Check that layer polygons exist
    if len(layer_polygons) < 1:
        logging.warning(f"No layer drawings found; will instead align to cortex depth")
        no_layers = True
    else:
        no_layers = False

    # Load the coordinates
    coord_df = pd.read_csv(coordinate_file)

    # Names of coordinate columns
    prefix = args["coordinate_column_prefix"]
    suffix = args["coordinate_column_suffix"]
    coord_cols = [prefix + c + suffix for c in ["x", "y", "z"]]

    # Determine the streamline field and upright angle

    # Construct strings from paths
    pia_path = ",".join(["{},{}".format(x, y) for x, y in pia_surface["path"]])
    wm_path = ",".join(["{},{}".format(x, y) for x, y in wm_surface["path"]])
    if soma_drawing is not None:
        soma_path = ",".join(["{},{}".format(x, y) for x, y in soma_drawing["path"]])
    else:
        soma_path = None
    resolution = pia_surface["resolution"]

    logging.info(f"Calculating depth field")
    depth_field, gradient_field, translation = run_streamlines(
        pia_path,
        wm_path,
        resolution,
        soma_path,
    )

    # Calculate rotation angle from the mean of the coordinates
    if soma_path is not None:
        logging.info("Using specified soma location in layer drawings file to determine upright angle")
        upright_angle = get_upright_angle(gradient_field)
    else:
        logging.info("No soma location specified in layer drawings file; using coordinate centroid to determine upright angle")
        upright_angle = get_upright_angle(
            gradient_field,
            coord_df[coord_cols].mean(axis=0).tolist(),
        )

    coords_to_transform = coord_df[coord_cols].values

    # Translate to correct location
    print(coords_to_transform[:5, :])
    coords_to_transform[:, :2] = coords_to_transform[:, :2] + translation
    print(coords_to_transform[:5, :])

    # get aligned y-values for coordinates
    if no_layers:
        logging.info("Calculating cortex-thickness adjusted depths for all points")
        y_values = cortex_thickness_aligned_y_values(
            coords_to_transform[:, 0], coords_to_transform[:, 1],
            avg_layer_depths, depth_field
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
        y_values = layer_aligned_y_values(
            coords_to_transform[:, 0], coords_to_transform[:, 1],
            avg_layer_depths, layer_list, depth_field, gradient_field, snapped_polys_surfs
        )


    # upright the coordinates
    rotation_upright_matrix = rotation_from_angle(upright_angle, axis=2)
    rotation_upright_affine = affine_from_transform_translation(
        transform=rotation_upright_matrix
    )
    T_rotate = AffineTransform(rotation_upright_affine)
    upright_coords = T_rotate.transform(coords_to_transform)

    # assign new y-values
    new_coord_df = coord_df.copy()
    new_coord_df[coord_cols] = upright_coords
    new_coord_df[coord_cols[1]] = y_values

    # save to cvs
    output_file = args["output_file"]
    new_coord_df.to_csv(output_file, index=False)


def console_script():
    module = ags.ArgSchemaParser(schema_type=LayerAlignedCoordsSchema)
    main(module.args)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=LayerAlignedCoordsSchema)
    main(module.args)
