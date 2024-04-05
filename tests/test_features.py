import unittest
import subprocess
import pandas as pd
import os
from importlib.resources import files

class TestFeatures(unittest.TestCase):

    def test_inhib_features(self):
            
        specimen_id_file = files('skeleton_keys') / "example_data/example_specimen_ids.txt"
        upright_swc_dir = files('skeleton_keys') / "example_data/upright_swcs"
        hist_file = files('skeleton_keys') / "example_data/aligned_depth_profiles.csv"
        soma_file = files('skeleton_keys') / "example_data/aligned_soma_depths.csv"
        long_output = files('skeleton_keys') / "example_data/PyTest_example_features_long.csv"
        axon_loading_file = files('skeleton_keys') / "example_data/PyTest_AxonLoadings.csv"
        
        command_args = {
            "specimen_id_file":specimen_id_file,
            "swc_dir":upright_swc_dir,
            "aligned_soma_file":soma_file,
            "aligned_depth_profile_file":hist_file,
            "analyze_axon":True,
            "analyze_basal_dendrite":True,
            "analyze_apical_dendrite":False,
            "save_axon_depth_profile_loadings_file":axon_loading_file,
            "output_file":long_output,
        }
        
        args_input = "".join([f" --{k} {v} " for k,v in command_args.items()])
        command = f"skelekeys-morph-features  {args_input}" 
        command_list = command.split()
        subprocess.run(command_list, check=True)

        wide_norm_ofile = files('skeleton_keys') / "example_data/PyTest_example_features_wide_normalized.csv"
        wide_unnorm_ofile = files('skeleton_keys') / "example_data/PyTest_example_features_wide_unnormalized.csv"

        feat_post_proc_cmd = 'skelekeys-postprocess-features --input_files "' + f"['{long_output}'" + f']" --wide_normalized_output_file {wide_norm_ofile} --wide_unnormalized_output_file {wide_unnorm_ofile}'
        os.system(feat_post_proc_cmd)
        
        file_equivalency = {
            long_output: files('skeleton_keys') / "example_data/example_features_long.csv",
            wide_norm_ofile: files('skeleton_keys') / "example_data/example_features_wide_normalized.csv",
            wide_unnorm_ofile: files('skeleton_keys') / "example_data/example_features_wide_unnormalized.csv",
            # axon_loading_file: files('skeleton_keys') / "example_data/axon_loadings.csv"
        }
        
        for test_file, static_file in file_equivalency.items():
            test_df = pd.read_csv(test_file,index_col=0)
            test_df = test_df.sort_values(by=test_df.columns.tolist()).reset_index(drop=True)
            
            static_df = pd.read_csv(static_file, index_col=0)
            static_df = static_df.sort_values(by=static_df.columns.tolist()).reset_index(drop=True)
            
            os.remove(test_file)
            self.assertIsNone(pd.testing.assert_frame_equal(test_df, static_df))

    def test_exc_features(self):
        
        # will test excitatory apical based features without providing a soma depth or histogram file
        
        specimen_id_file = files('skeleton_keys') / "example_data/example_specimen_excitatory_ids.txt"
        upright_swc_dir = files('skeleton_keys') / "example_data/upright_swcs"
        long_output = files('skeleton_keys') / "example_data/PyTest_example_excitatory_features_long.csv"

        command_args = {
            "specimen_id_file":specimen_id_file,
            "swc_dir":upright_swc_dir,
            "analyze_axon":False,
            "analyze_basal_dendrite":True,
            "analyze_apical_dendrite":True,
            "output_file":long_output,
        }
        
        args_input = "".join([f" --{k} {v} " for k,v in command_args.items()])
        command = f"skelekeys-morph-features  {args_input}" 
        command_list = command.split()
        subprocess.run(command_list, check=True)

        test_df = pd.read_csv(long_output,index_col=0)
        static_file_path = files('skeleton_keys') / "example_data/example_features_long_excitatory.csv"
        static_df = pd.read_csv(static_file_path, index_col=0)
    
        os.remove(long_output)
        self.assertIsNone(pd.testing.assert_frame_equal(test_df, static_df))
