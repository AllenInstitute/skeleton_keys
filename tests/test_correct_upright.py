import unittest
import numpy as np
from skeleton_keys.upright import upright_corrected_morph
from neuron_morphology.swc_io import Morphology


class TestUprightCorrect(unittest.TestCase):

    def setUp(self):
        self.morphology = Morphology([
            {'id': 1,
             "x": 0,
             "y": 0,
             "z": 0,
             "radius": 1,
             "type": 1,
             "parent": -1, },

            {'id': 2,
             "x": 2,
             "y": 0,
             "z": 1,
             "radius": 1,
             "type": 2,
             "parent": 1, }
        ],
            node_id_cb=lambda n: n['id'],
            parent_id_cb=lambda n: n['parent'])
        self.upright_angle = 90 * (np.pi / 180)
        self.slice_angle = 90 * (np.pi / 180)
        self.flip_status = -1
        self.shrink_factor = 5

    def test_correct_upright(self):
        morph = self.morphology
        upright_morph = upright_corrected_morph(morphology=morph,
                                                upright_angle=self.upright_angle,
                                                slice_angle=self.slice_angle,
                                                flip_status=self.flip_status,
                                                shrink_factor=self.shrink_factor)

        expected_coords = np.array([[0.0, 0.0, 0.0], [0.0, -5.0, 2.0]])
        upright_coords = np.array([[n['x'], n['y'], n['z']] for n in upright_morph.nodes()])

        self.assertIsNone(np.testing.assert_almost_equal(expected_coords, upright_coords))