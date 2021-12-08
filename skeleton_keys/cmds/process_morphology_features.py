import json
import argschema as ags
import numpy as np
import pandas as pd
from skeleton_keys.database_queries import swc_paths_from_database
from skeleton_keys.depth_profile import (
    calculate_pca_transforms_and_loadings,
    apply_loadings_to_profiles,
)


class ProcessMorphologyFeaturesParameters(ags.ArgSchema):
    specimen_id_file = ags.fields.InputFile(
        description="File with specimen IDs on each line",
    )
    swc_paths_file = ags.fields.InputFile(
        default=None,
        allow_none=True,
        description="optional - JSON file with swc file paths keyed on specimen IDs",
    )
    already_transformed = ags.fields.Boolean(
        default=False,
        description="Whether SWC files have already been transformed (uprighted, corrected for shrinkage & tilt)",
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
        descripton="List of layer names")
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
    long_unnormalized_output_file = ags.fields.OutputFile(
        default=None,
        allow_none=True,
        description="Long-form CSV of un-normalized features",
    )
    wide_normalized_output_file = ags.fields.OutputFile(
        default=None,
        allow_none=True,
        description="Wide-form CSV of normalized features",
    )
    wide_unnormalized_output_file = ags.fields.OutputFile(
        default=None,
        allow_none=True,
        description="Wide-form CSV of un-normalized features",
    )


def select_and_convert_depth_columns(df, prefix):
    """Select columns from depth profile DataFrame with particular prefix"""
    new_df = df.loc[:, df.columns.str.startswith(prefix)].copy()
    new_df.columns = [int(c.split(prefix)[-1]) for c in new_df.columns]
    sorted_columns = sorted(new_df.columns)
    return new_df.loc[:, sorted_columns]


def analyze_depth_profiles(df, preexisting_file, output_file):
    if preexisting_file is None:
        transformed, loadings = calculate_pca_transforms_and_loadings(
            df.values)
    else:
        loadings = pd.read_csv(preexisting_file, header=None).values
        transformed = apply_loadings_to_profiles(df.values, loadings)

    if output_file is not None:
        out_df = pd.DataFrame(loadings)
        out_df.to_csv(output_file, header=False, index=False)

    return transformed


def main():
    module = ags.ArgSchemaParser(schema_type=ProcessMorphologyFeaturesParameters)

    # Load specimen IDs
    specimen_id_file = module.args["specimen_id_file"]
    specimen_ids = np.loadtxt(specimen_id_file).astype(int)

    # Get paths to SWC files
    swc_paths_file = module.args["swc_paths_file"]
    if swc_paths_file is not None:
        with open(swc_paths_file, "r") as f:
            swc_paths = json.load(f)
        # ensure IDs are ints
        swc_paths = {int(k): v for k, v in swc_paths.items()}
    else:
        swc_paths = swc_paths_from_database(specimen_ids)

    # Load depth profiles
    aligned_depth_profile_file = module.args['aligned_depth_profile_file']
    depth_profile_df = pd.read_csv(aligned_depth_profile_file, index_col=0)

    # Compartment analysis flags
    analyze_axon_flag = module.args['analyze_axon']
    analyze_basal_flag = module.args['analyze_basal_dendrite']
    analyze_apical_flag = module.args['analyze_apical_dendrite']
    analyze_basal_dendrite_depth = module.args['analyze_basal_dendrite_depth']

    # Analyze depth profiles
    # Assumes that depth profile file has columns in the format:
    # "{compartment label}_{feature number}"

    depth_result = []
    if analyze_axon_flag:
        axon_depth_df = select_and_convert_depth_columns(depth_profile_df, "2_")
        available_ids = axon_depth_df.index.intersection(specimen_ids)
        transformed = analyze_depth_profiles(
            axon_depth_df.loc[available_ids, :],
            module.args["axon_depth_profile_loadings_file"],
            module.args["save_axon_depth_profile_loadings_file"]
        )
        for i, sp_id in enumerate(specimen_ids):
            for j in range(transformed.shape[1]):
                depth_result.append({
                    "specimen_id": sp_id,
                    "feature": f"depth_pc_{j}",
                    "compartment_type": "axon",
                    "dimension": "none",
                    "value": transformed[i, j],
                })

    if analyze_apical_flag:
        apical_depth_df = select_and_convert_depth_columns(depth_profile_df, "4_")
        available_ids = apical_depth_df.index.intersection(specimen_ids)
        transformed = analyze_depth_profiles(
            apical_depth_df.loc[available_ids, :],
            module.args["apical_dendrite_depth_profile_loadings_file"],
            module.args["save_apical_dendrite_depth_profile_loadings_file"]
        )
        for i, sp_id in enumerate(specimen_ids):
            for j in range(transformed.shape[1]):
                depth_result.append({
                    "specimen_id": sp_id,
                    "feature": f"depth_pc_{j}",
                    "compartment_type": "apical_dendrite",
                    "dimension": "none",
                    "value": transformed[i, j],
                })

    if analyze_basal_dendrite_depth:
        basal_depth_df = select_and_convert_depth_columns(depth_profile_df, "3_")
        available_ids = basal_depth_df.index.intersection(specimen_ids)
        transformed = analyze_depth_profiles(
            basal_depth_df.loc[available_ids, :],
            module.args["basal_dendrite_depth_profile_loadings_file"],
            module.args["save_basal_dendrite_depth_profile_loadings_file"]
        )
        for i, sp_id in enumerate(available_ids):
            for j in range(transformed.shape[1]):
                depth_result.append({
                    "specimen_id": sp_id,
                    "feature": f"depth_pc_{j}",
                    "compartment_type": "basal_dendrite",
                    "dimension": "none",
                    "value": transformed[i, j],
                })

    # TODO: ANALYZE PROFILE OVERLAP FEATURES

    # TODO: ANALYZE REST OF MORPH FEATURES

    # Save features to CSV

    long_unnormalized_output_file = module.args['long_unnormalized_output_file']
    if long_unnormalized_output_file is not None:
        long_result = []
        long_result += depth_result

        pd.DataFrame(long_result).to_csv(long_unnormalized_output_file)

    # TODO: SAVE WIDE FORMAT VERSIONS OF OUTPUTS


if __name__ == "__main__":
    main()