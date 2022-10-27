import json
import logging
import nrrd
import numpy as np
import argschema as ags
import skeleton_keys.full_morph as full_morph
import skeleton_keys.drawings as drawings

from neuron_morphology.swc_io import morphology_from_swc, morphology_to_swc
from allensdk.core.reference_space_cache import ReferenceSpaceCache


class FullMorphLayerDrawingsSchema(ags.ArgSchema):
    swc_path = ags.fields.InputFile(
        description="path to CCF-aligned full morphology SWC file"
    )
    atlas_volume_file = ags.fields.InputFile(
        description = "NRRD file of CCF atlas volume",
    )
    structure_list = ags.fields.List(
        ags.fields.String,
        cli_as_single_argument=True,
        description="List of structure acronyms to include around morphology",
    )
    closest_surface_voxel_file = ags.fields.InputFile(
        description="Closest surface voxel reference HDF5 file for slice angle calculation",
    )
    surface_paths_file = ags.fields.InputFile(
        description="Surface paths (streamlines) HDF5 file for slice angle calculation",
    )
    perimeter_simplify_tolerance = ags.fields.Float(
        description="tolerance parameter for perimeter simplification",
        default=2,
    )
    layer_simplify_tolerance = ags.fields.Float(
        description="tolerance parameter for layer border simplification",
        default=1,
    )
    output_drawings_file = ags.fields.OutputFile(
        description="Output JSON file with layer drawings",
    )
    output_swc_file = ags.fields.OutputFile(
        description="Output SWC file of morphology aligned with layer drawings")


def main(args):
    morph = morphology_from_swc(args["swc_path"])
    morph_soma = morph.get_soma()
    soma_coords = np.array([morph_soma['x'], morph_soma['y'], morph_soma['z']])

    atlas_volume, _ = nrrd.read(args["atlas_volume_file"])

    atlas_slice, angle_rad = full_morph.angled_atlas_slice_for_morph(
        morph,
        atlas_volume,
        args["closest_surface_voxel_file"],
        args["surface_paths_file"],
        return_angle=True,
    )

    rot_morph = full_morph.rotate_morphology_for_drawings(
        morph, angle_rad)

    reference_space_key = 'annotation/ccf_2017'
    resolution = 10
    rspc = ReferenceSpaceCache(resolution, reference_space_key, manifest='manifest.json')
    tree = rspc.get_structure_tree(structure_graph_id=1)

    atlas_slice = full_morph.select_structures_of_interest(
        atlas_slice,
        args["structure_list"],
        tree
    )
    atlas_slice = full_morph.merge_atlas_layers(atlas_slice, tree)

    atlas_slice = full_morph.remove_other_hemisphere(
        atlas_slice,
        soma_coords
    )

    boundaries = full_morph.find_layer_outlines(atlas_slice)
    perimeter = drawings.perimeter_of_layers(boundaries)
    simple_perimeter, corners = drawings.simplify_and_find_corners(
        perimeter,
        tolerance=args['perimeter_simplify_tolerance'],
        n_corners=4,
    )
    pia_side, wm_side = drawings.identify_pia_and_wm_sides(
        simple_perimeter,
        corners,
        boundaries,
    )

    # Prepare drawings for JSON file
    name_translation = {
        "Isocortex layer 1": "Layer1",
        "Isocortex layer 2/3": "Layer2/3",
        "Isocortex layer 4": "Layer4",
        "Isocortex layer 5": "Layer5",
        "Isocortex layer 6a": "Layer6a",
        "Isocortex layer 6b": "Layer6b",
    }

    rot_soma = rot_morph.get_soma()
    rot_soma_coords = [float(rot_soma['x']), float(rot_soma['y'])]
    drawing_data = {}
    drawing_data["pia_path"] = {
        "name": "Pia",
        "resolution": 1.0,
        "path": (np.array(pia_side.coords) * resolution).tolist(),
    }
    drawing_data["wm_path"] = {
        "name": "White Matter",
        "resolution": 1.0,
        "path": (np.array(wm_side.coords) * resolution).tolist(),
    }
    drawing_data["soma_path"] = {
        "name": "Soma",
        "resolution": 1.0,
        "path": [rot_soma_coords],
    }

    simplifed_boundaries = drawings.simplify_layer_boundaries(
        boundaries, args['layer_simplify_tolerance'])
    drawing_data['layer_polygons'] = []
    for n, b in simplifed_boundaries.items():
        drawing_data['layer_polygons'].append({
            "name": name_translation[n],
            "resolution": 1.0,
            "path": (b_coords * resolution).tolist(),
        })

    # Write output files
    with open(args['output_drawings_file'], 'w') as f:
        json.dump(drawing_data, f)

    morphology_to_swc(rot_morph, args['output_swc_file'])


def console_script():
    module = ags.ArgSchemaParser(schema_type=FullMorphLayerDrawingsSchema)
    main(module.args)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=FullMorphLayerDrawingsSchema)
    main(module.args)