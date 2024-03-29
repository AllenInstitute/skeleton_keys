from neuron_morphology.feature_extractor.marked_feature import (
    specialize, nested_specialize
)
from neuron_morphology.feature_extractor.feature_specialization import (
    NEURITE_SPECIALIZATIONS,
    AxonSpec,
    BasalDendriteSpec,
    ApicalDendriteSpec
)
from neuron_morphology.features.statistics.moments_along_max_distance_projection import \
    moments_along_max_distance_projection
from neuron_morphology.features.statistics.coordinates import COORD_TYPE_SPECIALIZATIONS
from neuron_morphology.features.dimension import dimension
from neuron_morphology.features.intrinsic import (
    num_branches,
    max_branch_order
)
from neuron_morphology.features.size import (
    total_length, total_surface_area, mean_diameter,
    max_euclidean_distance
)
from neuron_morphology.features.path import (
    max_path_distance, mean_contraction, early_branch_path
)
from neuron_morphology.features.soma import (
    soma_percentile, calculate_number_of_stems, calculate_stem_exit_and_distance, calculate_soma_surface
)
from neuron_morphology.features.branching.bifurcations import (
    num_outer_bifurcations
)


def default_features():
    """ Get set of default morphology features for feature extractor"""
    return [
        nested_specialize(
            dimension,
            [COORD_TYPE_SPECIALIZATIONS, NEURITE_SPECIALIZATIONS]),
        specialize(num_branches, NEURITE_SPECIALIZATIONS),
        specialize(max_branch_order, NEURITE_SPECIALIZATIONS),
        specialize(total_length, NEURITE_SPECIALIZATIONS),
        specialize(total_surface_area, NEURITE_SPECIALIZATIONS),
        specialize(mean_diameter, NEURITE_SPECIALIZATIONS),
        specialize(max_euclidean_distance, NEURITE_SPECIALIZATIONS),
        specialize(max_path_distance, NEURITE_SPECIALIZATIONS),
        specialize(mean_contraction, NEURITE_SPECIALIZATIONS),
        specialize(soma_percentile, NEURITE_SPECIALIZATIONS),
        specialize(calculate_number_of_stems, [BasalDendriteSpec]),
        specialize(calculate_stem_exit_and_distance, [AxonSpec, BasalDendriteSpec]),
        specialize(moments_along_max_distance_projection, [ApicalDendriteSpec]),
        specialize(early_branch_path, [ApicalDendriteSpec]),
        specialize(num_outer_bifurcations, [ApicalDendriteSpec]),
        calculate_soma_surface,
    ]
