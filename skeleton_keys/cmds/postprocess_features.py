import logging
import re
import numpy as np
import pandas as pd
import argschema as ags
from skeleton_keys import cloudfields
from skeleton_keys.io import read_csv, write_dataframe_to_csv


class PostprocessFeaturesParameters(ags.ArgSchema):
    input_files = ags.fields.List(
        cloudfields.InputFile(),
        cli_as_single_argument=True,
        description="Long-form CSV files of morphology features",
    )
    wide_normalized_output_file = cloudfields.OutputFile(
        description="Wide-form CSV of normalized features",
    )
    wide_unnormalized_output_file = cloudfields.OutputFile(
        description="Wide-form CSV of un-normalized features",
    )

    # Options of inclusions/exclusion
    drop_stem_exit = ags.fields.Boolean(
        default=False,
        description="Whether to drop stem exit features",
    )
    drop_bifurcation_angle = ags.fields.Boolean(
        default=False,
        description="Whether to drop bifurcation angle features",
    )
    drop_apical_n_stems = ags.fields.Boolean(
        default=False,
        description="Whether to drop number of apical dendrite stems features",
    )
    drop_neurite_radius_features = ags.fields.Boolean(
        default=False,
        description="Whether to drop features that are calculated using the radius of neurite compartments.",
    )
    drop_soma_surface_area = ags.fields.Boolean(
        default=False,
        description="Whether to drop the soma surface area features",
    )
    drop_nans = ags.fields.Boolean(
        default=False,
        description="Whether to drop cells that have nan for any values",
    )


def _natural_sort_key(s, _nsre=re.compile("([0-9]+)")):
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)
    ]


def main(args):
    df_list = []
    for filename in args["input_files"]:
        df = read_csv(filename, index_col=0)
        df_list.append(df)

    morph_df = pd.concat(df_list)

    drop_stem_exit = args["drop_stem_exit"]
    drop_bifurcation_angle = args["drop_bifurcation_angle"]
    drop_apical_n_stems = args["drop_apical_n_stems"]
    drop_neurite_radius_features = args["drop_neurite_radius_features"]
    drop_soma_surface_area = args["drop_soma_surface_area"]
    drop_nans = args["drop_nans"]

    if drop_stem_exit:
        mask = morph_df["feature"].str.startswith("stem_exit")
        morph_df = morph_df.loc[~mask, :]
    if drop_bifurcation_angle:
        mask = morph_df["feature"].str.startswith("bifurcation_angle")
        morph_df = morph_df.loc[~mask, :]
    if drop_apical_n_stems:
        mask = (morph_df["feature"].str.startswith("calculate_number_of_stems")) & (
            morph_df["compartment_type"] == "apical_dendrite"
        )
        morph_df = morph_df.loc[~mask, :]
    if drop_neurite_radius_features:
        mask = morph_df["feature"].str.startswith("mean_diameter") | morph_df[
            "feature"
        ].str.startswith("total_surface_area")
        morph_df = morph_df.loc[~mask, :]
    if drop_soma_surface_area:
        mask = (morph_df["feature"].str.startswith("surface_area")) & (
            morph_df["compartment_type"] == "soma"
        )
        morph_df = morph_df.loc[~mask, :]

    # log-transform number of outer bifurcations
    mask = morph_df["feature"].str.startswith("num_outer")
    morph_df.loc[mask, "value"] = np.log10(morph_df.loc[mask, "value"] + 1)

    morph_pt = (
        morph_df.set_index(["specimen_id", "feature", "compartment_type", "dimension"])
        .sort_index()
        .unstack(["feature", "compartment_type", "dimension"])
    )

    if drop_nans:
        # Find and drop cells that have nans for values
        null_rows = morph_pt.isnull().any(axis=1)
        print(null_rows)
        if np.sum(null_rows) > 0:
            logging.warning("Found cells with nan values; dropping cells")
            null_cols = morph_pt.isnull().any(axis=0)
            print(morph_pt.columns[null_cols])
            logging.warning(morph_pt.index[null_rows].tolist())
            morph_pt = morph_pt.loc[~null_rows, :]

    idx = pd.IndexSlice
    scales = {}
    for feat in morph_pt.columns.get_level_values("feature").unique():
        if "hist_pc" in feat:
            continue
        values = morph_pt.loc[:, idx[:, feat, :, :]].values
        scales[feat] = values.std()
        if scales[feat] == 0:  # zero std dev leads to nans
            scales[feat] = 1e-12

    morph_pt_norm = morph_pt.copy()
    for feat in morph_pt.columns.get_level_values("feature").unique():
        if feat in scales:
            morph_pt_norm.loc[:, idx[:, feat, :, :]] = (
                morph_pt.loc[:, idx[:, feat, :, :]]
                - morph_pt.loc[:, idx[:, feat, :, :]].mean()
            ) / scales[feat]
        else:
            morph_pt_norm.loc[:, idx[:, feat, :, :]] = (
                morph_pt.loc[:, idx[:, feat, :, :]]
                - morph_pt.loc[:, idx[:, feat, :, :]].mean()
            ) / morph_pt.loc[:, idx[:, feat, :, :]].std()

    labs = [
        "_".join(j)
        for j in zip(
            morph_pt.columns.get_level_values(2),
            morph_pt.columns.get_level_values(1),
            morph_pt.columns.get_level_values(3),
        )
    ]
    labs = [l.replace("_none", "") for l in labs]

    output_normalized_file = args["wide_normalized_output_file"]
    morph_pt_norm_export = morph_pt_norm.copy()
    morph_pt_norm_export.columns = labs
    morph_pt_norm_export = morph_pt_norm_export[sorted(labs, key=_natural_sort_key)]
    write_dataframe_to_csv(morph_pt_norm_export, output_normalized_file)

    output_raw_file = args["wide_unnormalized_output_file"]
    morph_pt_raw_export = morph_pt.copy()
    morph_pt_raw_export.columns = labs
    morph_pt_raw_export = morph_pt_raw_export[sorted(labs, key=_natural_sort_key)]
    write_dataframe_to_csv(morph_pt_raw_export, output_raw_file)


def console_script():
    module = ags.ArgSchemaParser(schema_type=PostprocessFeaturesParameters)
    main(module.args)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=PostprocessFeaturesParameters)
    main(module.args)
