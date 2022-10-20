from skeleton_keys.cmds.depth_profiles_from_aligned_swcs import (
    ProfilesFromAlignedSwcsParameters, 
    main as depth_profile_main
)

import argschema
from taskqueue import queueable


@queueable
def create_layer_histograms(
    specimen_id_file, 
    swc_dir, 
    layer_depths_file, 
    output_hist_file, 
    output_soma_file, 
    bin_size=5.0, 
    below_wm=200.0,):
    input_data = {
        "specimen_id_file": specimen_id_file,
        "swc_dir": swc_dir,
        "layer_depths_file": layer_depths_file,
        "output_hist_file": output_hist_file,
        "output_soma_file": output_soma_file,
        "bin_size": bin_size,
        "below_wm": below_wm,
    }

    input_data = {k: v for k, v in input_data.items() if v is not None}
    module = argschema.ArgSchemaParser(
        schema_type=ProfilesFromAlignedSwcsParameters, input_data=input_data, args=[]
    )
    depth_profile_main(module)