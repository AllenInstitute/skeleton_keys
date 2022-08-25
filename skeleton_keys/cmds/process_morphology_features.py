import json
import argschema as ags
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
from skeleton_keys.database_queries import (
    swc_paths_from_database,
    shrinkage_factor_from_database,
    pia_wm_soma_from_database,
)
from skeleton_keys.depth_profile import (
    calculate_pca_transforms_and_loadings,
    apply_loadings_to_profiles,
    earthmover_distance_between_compartments,
    overlap_between_compartments,
)
from skeleton_keys.feature_definition import default_features
from skeleton_keys.upright import upright_corrected_morph
from neuron_morphology.swc_io import morphology_from_swc
from neuron_morphology.feature_extractor.data import Data
from neuron_morphology.feature_extractor.feature_extractor import FeatureExtractor
from neuron_morphology.feature_extractor.utilities import unnest
from neuron_morphology.transforms.pia_wm_streamlines.calculate_pia_wm_streamlines import (
    run_streamlines,
)
from neuron_morphology.transforms.upright_angle.compute_angle import get_upright_angle
import os


class ProcessMorphologyFeaturesParameters(ags.ArgSchema):
    specimen_id_file = ags.fields.InputFile(
        description="File with specimen IDs on each line",
    )
    swc_paths_file = ags.fields.InputFile(
        default=None,
        allow_none=True,
        description="optional - JSON file with swc file paths keyed on specimen IDs",
    )
    swc_dir = ags.fields.InputDir(
        default=None,
        allow_none=True,
        description="optional - folder to find swc files, assuming specimen_id.swc is filename",
    )

    aligned_depth_profile_file = ags.fields.InputFile(
        description="CSV file with layer-aligned depth profile information",
    )
    aligned_soma_file = ags.fields.InputFile(
        description="CSV file with layer-aligned soma depth information",
    )
    layer_list = ags.fields.List(
        ags.fields.String,
        default=["1", "2/3", "4", "5", "6a", "6b"],
        cli_as_single_argument=True,
        descripton="List of layer names",
    )
    analyze_axon = ags.fields.Boolean(
        description="Whether to analyze axon compartments",
        default=False,
    )
    analyze_basal_dendrite = ags.fields.Boolean(
        description="Whether to analyze basal dendrite compartments",
        default=False,
    )
    analyze_apical_dendrite = ags.fields.Boolean(
        description="Whether to analyze apical dendrite compartments",
        default=False,
    )
    analyze_basal_dendrite_depth = ags.fields.Boolean(
        description="Whether to analyze depth profile of basal dendrite compartments",
        default=False,
    )
    axon_depth_profile_loadings_file = ags.fields.InputFile(
        default=None,
        allow_none=True,
        description="CSV with pre-existing axon depth profile loadings",
    )
    basal_dendrite_depth_profile_loadings_file = ags.fields.InputFile(
        default=None,
        allow_none=True,
        description="CSV with pre-existing basal dendrite depth profile loadings",
    )
    apical_dendrite_depth_profile_loadings_file = ags.fields.InputFile(
        default=None,
        allow_none=True,
        description="CSV with pre-existing apical dendrite depth profile loadings",
    )
    save_axon_depth_profile_loadings_file = ags.fields.OutputFile(
        default=None,
        allow_none=True,
        description="Output file to save axon depth profile loadings",
    )
    save_basal_dendrite_depth_profile_loadings_file = ags.fields.OutputFile(
        default=None,
        allow_none=True,
        description="Output file to save basal dendrite depth profile loadings",
    )
    save_apical_dendrite_depth_profile_loadings_file = ags.fields.OutputFile(
        default=None,
        allow_none=True,
        description="Output file to save apical dendrite depth profile loadings",
    )
    output_file = ags.fields.OutputFile(
        default=None,
        allow_none=True,
        description="Long-form CSV of un-normalized features",
    )


def soma_locations(df):
    """Obtain soma locations from DataFrame"""
    res = []
    for i, r in df.itertuples():
        res.append(
            {
                "specimen_id": i,
                "feature": "aligned_dist_from_pia",
                "compartment_type": "soma",
                "dimension": "none",
                "value": r,
            }
        )
    return res


def select_and_convert_depth_columns(df, prefix):
    """Select columns from depth profile DataFrame with particular prefix"""
    new_df = df.loc[:, df.columns.str.startswith(prefix)].copy()
    new_df.columns = [int(c.split(prefix)[-1]) for c in new_df.columns]
    sorted_columns = sorted(new_df.columns)
    return new_df.loc[:, sorted_columns]


def analyze_depth_profiles(df, preexisting_file, output_file):
    if preexisting_file is None:
        transformed, loadings = calculate_pca_transforms_and_loadings(df.values)
    else:
        loadings = pd.read_csv(preexisting_file, header=None).values
        transformed = apply_loadings_to_profiles(df.values, loadings)

    if output_file is not None:
        out_df = pd.DataFrame(loadings)
        out_df.to_csv(output_file, header=False, index=False)

    return transformed


def specimen_morph_features(
    specimen_id,
    swc_path,
    layer_list,
    analyze_axon,
    analyze_apical_dendrite,
    analyze_basal_dendrite,
):

    # Load the morphology and transform if necessary
    morph = morphology_from_swc(swc_path)

    cell_data = Data(morphology=morph)
    fe = FeatureExtractor()
    fe.register_features(default_features())
    fe_results = fe.extract(cell_data)

    # Determine compartments from which to keep features
    compartments = ["soma"]
    if analyze_axon:
        compartments.append("axon")
    if analyze_basal_dendrite:
        compartments.append("basal_dendrite")
    if analyze_apical_dendrite:
        compartments.append("apical_dendrite")

    # Define logic for which features to keep or handle specially
    unchanged_features = [
        "num_branches",
        "max_branch_order",
        "total_length",
        "max_euclidean_distance",
        "max_path_distance",
        "mean_contraction",
    ]
    dendrite_only_features = [
        "total_surface_area",
        "mean_diameter",
        "calculate_number_of_stems",
    ]

    # Unpack the data structure from the neuron_morphology feature extractor
    # to put features in format for output; handle special cases
    result_list = []
    long_results = unnest(fe_results.results)
    for fullname, value in long_results.items():
        split_name = fullname.split(".")
        compartment_name = split_name[0]

        # neuron_morphology assigns the calculate_soma_surface.name attribute to 'calculate_soma_surface' so
        # split_name and compartment_name are = 'calculate_soma_surface'
        if not any([c in compartment_name for c in compartments]):
            continue
        if len(split_name) > 1:
            primary_feature = split_name[1]
        else:
            primary_feature = split_name[0]

        result = {
            "specimen_id": specimen_id,
            "feature": primary_feature,
            "compartment_type": compartment_name,
        }
        if primary_feature in unchanged_features:
            result["dimension"] = "none"
            result["value"] = value
            result_list.append(result)
        elif primary_feature == "calculate_soma_surface":
            result["dimension"] = "none"
            result["value"] = value
            result["feature"] = "surface_area"
            result["compartment_type"] = "soma"
            result_list.append(result)
        elif primary_feature == "soma_percentile":
            # soma percentile returned as a 2-tuple; split them into
            # separate entries for x & y
            result_x = result.copy()
            result_x["dimension"] = "x"
            result_x["value"] = value[0]

            result_y = result.copy()
            result_y["dimension"] = "y"
            result_y["value"] = value[1]
            result_list += [result_x, result_y]
        elif primary_feature in dendrite_only_features and compartment_name in (
            "basal_dendrite",
            "apical_dendrite",
        ):
            result["dimension"] = "none"
            result["value"] = value
            result_list.append(result)
        elif primary_feature == "node":
            if len(split_name) >= 4:
                if split_name[2] == "dimension" and split_name[3] == "bias_xyz":
                    result["feature"] = "bias"
                    result_x = result.copy()
                    result_x["dimension"] = "x"
                    result_x["value"] = abs(value[0])

                    result_y = result.copy()
                    result_y["dimension"] = "y"
                    result_y["value"] = value[1]
                    result_list += [result_x, result_y]
                elif split_name[2] == "dimension" and split_name[3] == "width":
                    result["feature"] = "extent"
                    result["dimension"] = "x"
                    result["value"] = value
                    result_list.append(result)
                elif split_name[2] == "dimension" and split_name[3] == "height":
                    result["feature"] = "extent"
                    result["dimension"] = "y"
                    result["value"] = value
                    result_list.append(result)

    if "basal_dendrite.calculate_stem_exit_and_distance" in long_results:
        stem_info = long_results["basal_dendrite.calculate_stem_exit_and_distance"]
        total_stems = len(stem_info)
        down = 0
        side = 0
        up = 0
        for theta, distance in stem_info:
            if theta <= 1 / 3:
                down += 1
            elif theta <= 2 / 3:
                side += 1
            else:
                up += 1
        for val, dim in zip((down, side, up), ("down", "side", "up")):
            result_list.append(
                {
                    "specimen_id": specimen_id,
                    "feature": "stem_exit",
                    "compartment_type": "basal_dendrite",
                    "dimension": dim,
                    "value": val / total_stems,
                }
            )

    if "axon.calculate_stem_exit_and_distance" in long_results:
        # find closest one
        closest_distance = np.inf
        closest_theta = 0
        for theta, distance in long_results["axon.calculate_stem_exit_and_distance"]:
            if distance < closest_distance:
                closest_distance = distance
                closest_theta = theta

        result_list.append(
            {
                "specimen_id": specimen_id,
                "feature": "exit_theta",
                "compartment_type": "axon",
                "dimension": "none",
                "value": closest_theta,
            }
        )
        result_list.append(
            {
                "specimen_id": specimen_id,
                "feature": "exit_distance",
                "compartment_type": "axon",
                "dimension": "none",
                "value": closest_distance,
            }
        )

    return result_list


def main(args):
    # Load specimen IDs
    specimen_id_file = args["specimen_id_file"]
    specimen_ids = np.loadtxt(specimen_id_file, ndmin=1).astype(int)

    # Get paths to SWC files
    swc_paths_file = args["swc_paths_file"]
    swc_dir = args["swc_dir"]
    if swc_paths_file is not None:
        with open(swc_paths_file, "r") as f:
            swc_paths = json.load(f)
        # ensure IDs are ints
        swc_paths = {int(k): v for k, v in swc_paths.items()}
    elif swc_dir is not None:
        swc_paths = {k: os.path.join(swc_dir, f"{k}.swc") for k in specimen_ids}
    else:
        swc_paths = swc_paths_from_database(specimen_ids)

    # Load soma depths
    aligned_soma_file = args["aligned_soma_file"]
    soma_loc_df = pd.read_csv(aligned_soma_file, index_col=0)
    soma_loc_res = soma_locations(
        soma_loc_df.loc[soma_loc_df.index.intersection(specimen_ids), :]
    )

    # Load depth profiles
    aligned_depth_profile_file = args["aligned_depth_profile_file"]
    depth_profile_df = pd.read_csv(aligned_depth_profile_file, index_col=0)

    # Compartment analysis flags
    analyze_axon_flag = args["analyze_axon"]
    analyze_basal_flag = args["analyze_basal_dendrite"]
    analyze_apical_flag = args["analyze_apical_dendrite"]
    analyze_basal_dendrite_depth_flag = args["analyze_basal_dendrite_depth"]

    # Analyze depth profiles
    # Assumes that depth profile file has columns in the format:
    # "{compartment label}_{feature number}"

    depth_result = []
    axon_depth_df = None
    apical_depth_df = None
    basal_depth_df = None

    if analyze_axon_flag:
        axon_depth_df = select_and_convert_depth_columns(depth_profile_df, "2_")
        available_ids = axon_depth_df.index.intersection(specimen_ids)
        transformed = analyze_depth_profiles(
            axon_depth_df.loc[available_ids, :],
            args["axon_depth_profile_loadings_file"],
            args["save_axon_depth_profile_loadings_file"],
        )
        for i, sp_id in enumerate(specimen_ids):
            for j in range(transformed.shape[1]):
                depth_result.append(
                    {
                        "specimen_id": sp_id,
                        "feature": f"depth_pc_{j}",
                        "compartment_type": "axon",
                        "dimension": "none",
                        "value": transformed[i, j],
                    }
                )

    if analyze_apical_flag:
        apical_depth_df = select_and_convert_depth_columns(depth_profile_df, "4_")
        available_ids = apical_depth_df.index.intersection(specimen_ids)
        transformed = analyze_depth_profiles(
            apical_depth_df.loc[available_ids, :],
            args["apical_dendrite_depth_profile_loadings_file"],
            args["save_apical_dendrite_depth_profile_loadings_file"],
        )
        for i, sp_id in enumerate(specimen_ids):
            for j in range(transformed.shape[1]):
                depth_result.append(
                    {
                        "specimen_id": sp_id,
                        "feature": f"depth_pc_{j}",
                        "compartment_type": "apical_dendrite",
                        "dimension": "none",
                        "value": transformed[i, j],
                    }
                )

    if analyze_basal_flag:
        basal_depth_df = select_and_convert_depth_columns(depth_profile_df, "3_")

        # Extra option, since PCA on basal dendrite profiles is not
        # typically that interesting, but we need the basal_depth_df for
        # other analyses
        if analyze_basal_dendrite_depth_flag:
            available_ids = basal_depth_df.index.intersection(specimen_ids)
            transformed = analyze_depth_profiles(
                basal_depth_df.loc[available_ids, :],
                args["basal_dendrite_depth_profile_loadings_file"],
                args["save_basal_dendrite_depth_profile_loadings_file"],
            )
            for i, sp_id in enumerate(available_ids):
                for j in range(transformed.shape[1]):
                    depth_result.append(
                        {
                            "specimen_id": sp_id,
                            "feature": f"depth_pc_{j}",
                            "compartment_type": "basal_dendrite",
                            "dimension": "none",
                            "value": transformed[i, j],
                        }
                    )

    # determine pairs of compartments to compare
    profile_comparison_pairs = []
    if analyze_axon_flag and analyze_apical_flag:
        profile_comparison_pairs.append(
            ("axon", "apical_dendrite", axon_depth_df, apical_depth_df)
        )
    if analyze_axon_flag and analyze_basal_flag:
        profile_comparison_pairs.append(
            ("axon", "basal_dendrite", axon_depth_df, basal_depth_df)
        )
    if analyze_apical_flag and analyze_basal_flag:
        profile_comparison_pairs.append(
            ("apical_dendrite", "basal_dendrite", apical_depth_df, basal_depth_df)
        )

    # Analyze earthmover distances and overlap between depth profiles
    profile_comparison_result = []
    for name_a, name_b, df_a, df_b in profile_comparison_pairs:
        emd_df = earthmover_distance_between_compartments(df_a, df_b)
        for r in emd_df.itertuples():
            profile_comparison_result.append(
                {
                    "specimen_id": r.Index,
                    "feature": f"emd_with_{name_b}",
                    "compartment_type": name_a,
                    "dimension": "none",
                    "value": r.emd,
                }
            )
        overlap_a_b_df = overlap_between_compartments(df_a, df_b)
        for r in overlap_a_b_df.itertuples():
            r_dict = r._asdict()
            for feature in ("frac_above", "frac_intersect", "frac_below"):
                profile_comparison_result.append(
                    {
                        "specimen_id": r.Index,
                        "feature": f"{feature}_{name_b}",
                        "compartment_type": name_a,
                        "dimension": "none",
                        "value": r_dict[feature],
                    }
                )
        overlap_b_a_df = overlap_between_compartments(df_b, df_a)
        for r in overlap_b_a_df.itertuples():
            for feature in ("frac_above", "frac_intersect", "frac_below"):
                r_dict = r._asdict()
                profile_comparison_result.append(
                    {
                        "specimen_id": r.Index,
                        "feature": f"{feature}_{name_a}",
                        "compartment_type": name_b,
                        "dimension": "none",
                        "value": r_dict[feature],
                    }
                )

    # Analyze rest of morphological features of cell
    map_input = [
        (
            specimen_id,
            swc_paths[specimen_id],
            args["layer_list"],
            analyze_axon_flag,
            analyze_apical_flag,
            analyze_basal_flag,
        )
        for specimen_id in specimen_ids
    ]
    pool = Pool()
    morph_results = pool.starmap(specimen_morph_features, map_input)

    # Save features to CSV
    long_result = []
    long_result += soma_loc_res
    long_result += depth_result
    long_result += profile_comparison_result
    for res in morph_results:
        long_result += res

    output_file = args["output_file"]
    pd.DataFrame(long_result).to_csv(output_file)


def console_script():
    module = ags.ArgSchemaParser(schema_type=ProcessMorphologyFeaturesParameters)
    main(module.args)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=ProcessMorphologyFeaturesParameters)
    main(module.args)
