import json
import argschema as ags
import numpy as np
import pandas as pd
from skeleton_keys.database_queries import swc_paths_from_database

from skeleton_keys.cmds.process_morphology_features import (
    select_and_convert_depth_columns,
    analyze_depth_profiles,
)
import os


class CalcHistogramLoadingsParameters(ags.ArgSchema):
    specimen_id_file = ags.fields.InputFile(
        description="File with specimen IDs on each line",
    )
    aligned_depth_profile_file = ags.fields.InputFile(
        description="CSV file with layer-aligned depth profile information",
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


def main(args):
    # Load specimen IDs
    specimen_id_file = args["specimen_id_file"]
    specimen_ids = np.loadtxt(specimen_id_file, ndmin=1).astype(int)

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

    axon_depth_df = None
    apical_depth_df = None
    basal_depth_df = None

    if analyze_axon_flag:
        axon_depth_df = select_and_convert_depth_columns(depth_profile_df, "2_")
        available_ids = axon_depth_df.index.intersection(specimen_ids)
        transformed = analyze_depth_profiles(
            axon_depth_df.loc[available_ids, :],
            None,
            args["save_axon_depth_profile_loadings_file"],
        )

    if analyze_apical_flag:
        apical_depth_df = select_and_convert_depth_columns(depth_profile_df, "4_")
        available_ids = apical_depth_df.index.intersection(specimen_ids)
        transformed = analyze_depth_profiles(
            apical_depth_df.loc[available_ids, :],
            None,
            args["save_apical_dendrite_depth_profile_loadings_file"],
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
                None,
                args["save_basal_dendrite_depth_profile_loadings_file"],
            )


def console_script():
    module = ags.ArgSchemaParser(schema_type=CalcHistogramLoadingsParameters)
    main(module.args)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=CalcHistogramLoadingsParameters)
    main(module.args)
