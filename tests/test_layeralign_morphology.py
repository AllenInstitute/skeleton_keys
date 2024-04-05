import unittest
import numpy as np
from neuron_morphology.swc_io import morphology_from_swc
import os
import subprocess
from neuron_morphology.transforms.affine_transform import (
    affine_from_translation,
    AffineTransform,
)
from importlib.resources import files


def are_dicts_equal(dict1, dict2):
    """
    Check if two dictionaries are equal.
    """
    if len(dict1) != len(dict2):
        print("Nodes have different number of key/value pairs")
        return False

    for key in dict1:
        if key not in dict2 or round(float(dict1[key]),3) != round(float(dict2[key]),3):            
            # print(round(dict1[key],3) , round(dict2[key],3))
            return False
    
    return True

def morphology_equivalence(morph1, morph2):
    """
    Check if two lists of dictionaries are equal.
    """
    if len(morph1) != len(morph2):
        print("morphologys have different number of nodes")
        return False

    for item1, item2 in zip(morph1.nodes(), morph2.nodes()):
        if not are_dicts_equal(item1, item2):
            print("Morphologies are not identical. The nodes should appear in the same order, and should be equivalent")
            print(item1)
            print(item2)
            return False

    return True


class TestLayerAlign(unittest.TestCase):

    def setUp(self):
        self.id_to_specimen_name = {
            606271263: 'Vipr2-IRES2-Cre_Slc32a1-T2A-FlpO_Ai65-337416.05.01.01_678210885_m.swc',
            694146546: 'Vip-IRES-Cre_Ai14-387150.05.02.01_811371030_m.swc',
            740135032: 'Sst-IRES-Cre_Ai14-408415.03.02.02_864876723_m.swc'
            }
        
    def test_layer_align(self):
        for specimen_id, specimen_name in self.id_to_specimen_name.items():
            
            raw_swc = files('skeleton_keys') / "example_data/{}".format(specimen_name)
            layer_drawings_file = files('skeleton_keys') / "example_data/{}_surfaces_and_layers.json".format(specimen_id)
            temp_output = str(files('skeleton_keys') / "example_data/{}_PytestLayerAlign.swc".format(specimen_id))
            
            align_args = {
                "specimen_id":specimen_id,
                "swc_path":raw_swc,
                "output_file":temp_output,
                "correct_for_shrinkage":False,
                "correct_for_slice_angle":False,
                "surface_and_layers_file":layer_drawings_file,    
            }
            args_input = "".join([f" --{k} {v} " for k,v in align_args.items()])
            layeralign_command = f"skelekeys-layer-aligned-swc {args_input}" 
            command_list = layeralign_command.split()
            subprocess.run(command_list, check=True)

            precalculate_swc = str(files('skeleton_keys') / "example_data/layer_aligned_swcs/{}.swc".format(specimen_id))
            precalculated_morph = morphology_from_swc(precalculate_swc)
            
            test_morph = morphology_from_swc(temp_output)
            
            # the precalculated swc file was not centered so we need to align the somas
            current_soma = test_morph.get_soma()
            previous_soma = precalculated_morph.get_soma()

            translation_for_soma = np.array([current_soma[i]-previous_soma[i] for i in ['x','y','z']])
            translation_affine = affine_from_translation(translation_for_soma)
            T_translate = AffineTransform(translation_affine)
            T_translate.transform_morphology(precalculated_morph)
            
            morph_equiv = morphology_equivalence(test_morph, precalculated_morph)
            os.remove(temp_output)
            self.assertTrue(morph_equiv)
            