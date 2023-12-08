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
    base_orientation = ags.fields.String(
        default=None,
        allow_none=True,
        description='initial orientation of slice before tilting (if used). If option is None, the slice will be determined by finding the direction of minimum curvature near the cell. "auto" selects the option with the smaller tilt. otherwise, a coronal or parasagittal slice can be specified',
        validation=lambda x: x in [None, 'auto', 'coronal', 'parasagittal'],
    )
    closest_surface_voxel_file = ags.fields.InputFile(
        description="Closest surface voxel reference HDF5 file for slice angle calculation",
    )
    surface_paths_file = ags.fields.InputFile(
        description="Surface paths (streamlines) HDF5 file for slice angle calculation",
    )
    pia_curvature_surface_file = ags.fields.InputFile(
        description="VTP file path for pia surface curvature values",
        allow_none=True,
    )
    wm_curvature_surface_file = ags.fields.InputFile(
        description="VTP file path for white matter surface curvature values",
        allow_none=True,
    )
    perimeter_simplify_tolerance = ags.fields.Float(
        description="tolerance parameter for perimeter simplification",
        default=2,
    )
    layer_simplify_tolerance = ags.fields.Float(
        description="tolerance parameter for layer border simplification",
        default=1,
    )
    min_contour_pts = ags.fields.Integer(
        description="minimum number of points to consider layer boundary contour",
        default=10,
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

    reference_space_key = 'annotation/ccf_2017'
    resolution = 10
    rspc = ReferenceSpaceCache(resolution, reference_space_key, manifest='manifest.json')
    tree = rspc.get_structure_tree(structure_graph_id=1)
    acronym_map = rspc.get_reference_space().structure_tree.get_id_acronym_map()
    base_orientation = args['base_orientation']

    # Check if the cells soma is out of cortex, if it is we will use the nearest isocortex voxel in place of its soma
    isocortex_struct_id = acronym_map['Isocortex']
    out_of_cortex_bool, nearest_cortex_coord = full_morph.check_coord_out_of_cortex(soma_coords,
                                                                                    isocortex_struct_id,
                                                                                    atlas_volume,
                                                                                    args['closest_surface_voxel_file'],
                                                                                    args['surface_paths_file'],
                                                                                    tree)
    
    if base_orientation is None:
        atlas_slice, q = full_morph.min_curvature_atlas_slice_for_morph(
            morph,
            atlas_volume,
            args["closest_surface_voxel_file"],
            args["surface_paths_file"],
            args["pia_curvature_surface_file"],
            args["wm_curvature_surface_file"],
            nearest_cortex_coord
        )

        rot_morph, rot_nearest_cortex_coord = full_morph.rotate_morphology_for_drawings_by_rotation(
            morph, q, nearest_cortex_coord)
        rot_morph, rot_nearest_cortex_coord = full_morph.align_morphology_to_drawings(rot_morph, atlas_slice,
                                                                                      rot_nearest_cortex_coord)
    else:
        atlas_slice, angle_rad, base_orientation = full_morph.angled_atlas_slice_for_morph(
            morph,
            atlas_volume,
            args["closest_surface_voxel_file"],
            args["surface_paths_file"],
            base_orientation=base_orientation,
            return_angle=True,
            closest_cortex_node=nearest_cortex_coord,
        )
        rot_morph, rot_nearest_cortex_coord = full_morph.rotate_morphology_for_drawings_by_angle(
            morph, angle_rad, base_orientation, nearest_cortex_coord)

    atlas_slice = full_morph.select_structures_of_interest(
        atlas_slice,
        args["structure_list"],
        tree
    )
    atlas_slice = full_morph.merge_atlas_layers(atlas_slice, tree)

    if base_orientation is not None and base_orientation == 'coronal':
        atlas_slice = full_morph.remove_other_hemisphere(
            atlas_slice,
            soma_coords
        )

    drawing_soma = rot_morph.get_soma()
    drawing_soma_coords = np.array([drawing_soma['x'], drawing_soma['y'], drawing_soma['z']])

    boundaries = full_morph.find_layer_outlines(
        atlas_slice, drawing_soma_coords, min_contour_pts=args['min_contour_pts'])
    perimeter = drawings.perimeter_of_layers(boundaries)
    simple_perimeter, pia_side, wm_side = drawings.simplify_and_find_sides(
        perimeter,
        boundaries,
        tolerance=args['perimeter_simplify_tolerance'],
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
    
    # If the soma was out of cortex we will pass in the nearest cortex node to the drawings output file
    if out_of_cortex_bool:
        soma_drawing_data = [float(rot_nearest_cortex_coord[0]), float(rot_nearest_cortex_coord[1])]
    else:
        rot_soma = rot_morph.get_soma()
        soma_drawing_data = [float(rot_soma['x']), float(rot_soma['y'])]

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
        "path": [soma_drawing_data],
    }

    simplifed_boundaries = drawings.simplify_layer_boundaries(
        boundaries, args['layer_simplify_tolerance'])
    drawing_data['layer_polygons'] = []
    for n, b in simplifed_boundaries.items():
        drawing_data['layer_polygons'].append({
            "name": name_translation[n],
            "resolution": 1.0,
            "path": (b * resolution).tolist(),
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