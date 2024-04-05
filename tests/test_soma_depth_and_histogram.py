import unittest
import subprocess
import pandas as pd
import os
from importlib.resources import files

class TestSomaDepthAndHistogram(unittest.TestCase):

    def test_histogram_and_soma_depth(self):
        
        specimen_id_file = files('skeleton_keys') / "example_data/example_specimen_ids.txt"
        layer_aligned_swc_dir = files('skeleton_keys') / "example_data/layer_aligned_swcs"
        output_hist_file = files('skeleton_keys') / "example_data/PyTest_AlignedDepthProfiles.csv"
        output_soma_file = files('skeleton_keys') / "example_data/PyTest_AlignedSomadepths.csv"
        
        command_args = {
            "specimen_id_file":specimen_id_file,
            "swc_dir":layer_aligned_swc_dir,
            "output_hist_file":output_hist_file,
            "output_soma_file":output_soma_file,
        }
        args_input = "".join([f" --{k} {v} " for k,v in command_args.items()])
        command = f"skelekeys-profiles-from-swcs {args_input}" 
        command_list = command.split()
        subprocess.run(command_list, check=True)

        static_somas_file = files('skeleton_keys') / "example_data/aligned_soma_depths.csv"
        test_somas = pd.read_csv(output_soma_file, encoding='utf-8') 
        static_somas = pd.read_csv(static_somas_file, encoding='utf-8')

        os.remove(output_soma_file)
        self.assertIsNone(pd.testing.assert_frame_equal(test_somas, static_somas))
        
        test_hists = pd.read_csv(output_hist_file)
        static_hist_file = files('skeleton_keys') / "example_data/aligned_depth_profiles.csv"
        static_hists = pd.read_csv(static_hist_file)
        os.remove(output_hist_file)
        self.assertIsNone(pd.testing.assert_frame_equal(test_hists, static_hists))

        