from skeleton_keys.cmds.depth_profiles_from_aligned_swcs import (
    ProfilesFromAlignedSwcsParameters, 
    main as depth_profile_main
)

from skeleton_keys.cmds.process_morphology_features import (
    ProcessMorphologyFeaturesParameters,
    main as morph_feat_main,
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

@queueable
def extract_morphology_features(
    specimen_id_file,
    aligned_depth_profile_file,
    aligned_soma_file,
    output_file,
    swc_dir=None,
    swc_paths_file=None,
    layer_list=None,
    analyze_axon=False,
    analyze_basal_dendrite=False,
    analyze_apical_dendrite=False,
    analyze_basal_dendrite_depth=False,
    axon_depth_profile_loadings_file=None,
    basal_dendrite_depth_profile_loadings_file=None,
    apical_dendrite_depth_profile_loadings_file=None,
):
    input_data = {
        "specimen_id_file": specimen_id_file,
        "aligned_depth_profile_file": aligned_depth_profile_file,
        "aligned_soma_file": aligned_soma_file,
        "swc_dir": swc_dir,
        "swc_paths_file": swc_paths_file,
        "layer_list": layer_list,
        "analyze_axon": analyze_axon,
        "analyze_basal_dendrite": analyze_basal_dendrite,
        "analyze_apical_dendrite": analyze_apical_dendrite,
        "analyze_basal_dendrite_depth": analyze_basal_dendrite_depth,
        "axon_depth_profile_loadings_file": axon_depth_profile_loadings_file,
        "basal_dendrite_depth_profile_loadings_file": basal_dendrite_depth_profile_loadings_file,
        "apical_dendrite_depth_profile_loadings_file": apical_dendrite_depth_profile_loadings_file,
        "output_file": output_file,
    }

    input_data = {k: v for k, v in input_data.items() if v is not None}
    module = argschema.ArgSchemaParser(
        schema_type=ProcessMorphologyFeaturesParameters, input_data=input_data, args=[]
    )
    morph_feat_main(module.args)