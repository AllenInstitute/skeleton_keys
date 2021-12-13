from taskqueue import queueable
from skeleton_keys.cmds.layer_aligned_swc import (
    LayerAlignedSwcSchema,
    main as layer_align_main,
)
from skeleton_keys.cmds.upright_corrected_swc import (
    UprightCorrectedSwcSchema,
    main as upright_main,
)
import argschema


@queueable
def layer_align_cell(
    specimen_id,
    swc_path,
    layer_depths_file,
    output_file,
    surface_and_layers_file,
    correct_for_shrinkage=False,
    correct_for_slice_angle=False,
    closest_surface_voxel_file=None,
    surface_paths_file=None,
    layer_list=None,
):

    input_data = {
        "specimen_id": specimen_id,
        "swc_path": swc_path,
        "layer_depths_file": layer_depths_file,
        "output_file": output_file,
        "correct_for_shrinkage": correct_for_shrinkage,
        "correct_for_slice_angle": correct_for_slice_angle,
        "surface_and_layers_file": surface_and_layers_file,
        "closest_surface_voxel_file": closest_surface_voxel_file,
        "surface_paths_file": surface_paths_file,
        "layer_list": layer_list,
    }
    input_data = {k: v for k, v in input_data.items() if v is not None}
    module = argschema.ArgSchemaParser(
        schema_type=LayerAlignedSwcSchema, input_data=input_data, args=[]
    )
    layer_align_main(module.args)


@queueable
def upright_correct_cell(
    specimen_id,
    swc_path,
    output_file,
    surface_and_layers_file,
    correct_for_shrinkage=False,
    correct_for_slice_angle=False,
    closest_surface_voxel_file=None,
    surface_paths_file=None,
):
    input_data = {
        "specimen_id": specimen_id,
        "swc_path": swc_path,
        "output_file": output_file,
        "correct_for_shrinkage": correct_for_shrinkage,
        "correct_for_slice_angle": correct_for_slice_angle,
        "surface_and_layers_file": surface_and_layers_file,
        "closest_surface_voxel_file": closest_surface_voxel_file,
        "surface_paths_file": surface_paths_file,
    }
    input_data = {k: v for k, v in input_data.items() if v is not None}
    module = argschema.ArgSchemaParser(
        schema_type=UprightCorrectedSwcSchema, input_data=input_data, args=[]
    )
    upright_main(module.args)
