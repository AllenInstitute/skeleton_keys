import logging
import re
import numpy as np
import pandas as pd
import argschema as ags


class PostprocessFeaturesParameters(ags.ArgSchema):
    input_files = ags.fields.List(
        ags.fields.InputFile(),
        cli_as_single_argument=True,
        description="Long-form CSV files of morphology features",
    )
    wide_normalized_output_file = ags.fields.OutputFile(
        description="Wide-form CSV of normalized features",
    )
    wide_unnormalized_output_file = ags.fields.OutputFile(
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


def _natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


def main():
    module = ags.ArgSchemaParser(schema_type=PostprocessFeaturesParameters)

    df_list = []
    for filename in module.args["input_files"]:
        df = pd.read_csv(filename, index_col=0)
        df_list.append(df)

    morph_df = pd.concat(df_list)

    drop_stem_exit = module.args["drop_stem_exit"]
    drop_bifurcation_angle = module.args["drop_bifurcation_angle"]
    drop_apical_n_stems = module.args["drop_apical_n_stems"]

    if drop_stem_exit:
        mask = ((morph_df["feature"].str.startswith("stem_exit")))
        morph_df = morph_df.loc[~mask, :]
    if drop_bifurcation_angle:
        mask = morph_df["feature"].str.startswith("bifurcation_angle")
        morph_df = morph_df.loc[~mask, :]
    if drop_apical_n_stems:
        mask = ((morph_df["feature"].str.startswith("calculate_number_of_stems")) &
            (morph_df["compartment_type"] == "apical_dendrite"))
        morph_df = morph_df.loc[~mask, :]

    # log-transform number of outer bifurcations
    mask = ((morph_df["feature"].str.startswith("num_outer")))
    morph_df.loc[mask, "value"] = np.log10(morph_df.loc[mask, "value"] + 1)

    morph_pt = morph_df.set_index(["specimen_id",
                                   "feature",
                                   "compartment_type",
                                   "dimension"]).sort_index().unstack(["feature", "compartment_type", "dimension"])

    print(morph_pt.head())

    # Find and drop cells that have nans for values
    null_rows = morph_pt.isnull().any(axis=1)
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
        print(feat, len(values))
        scales[feat] = values.std()
        if scales[feat] == 0: # zero std dev leads to nans
            scales[feat] = 1e-12

    morph_pt_norm = morph_pt.copy()
    for feat in morph_pt.columns.get_level_values("feature").unique():
        if feat in scales:
            morph_pt_norm.loc[:, idx[:, feat, :, :]] = (morph_pt.loc[:, idx[:, feat, :, :]] - morph_pt.loc[:, idx[:, feat, :, :]].mean()) / scales[feat]
        else:
            morph_pt_norm.loc[:, idx[:, feat, :, :]] = (morph_pt.loc[:, idx[:, feat, :, :]] - morph_pt.loc[:, idx[:, feat, :, :]].mean()) / morph_pt.loc[:, idx[:, feat, :, :]].std()

    labs = ["_".join(j) for j in zip(morph_pt.columns.get_level_values(2),
                                     morph_pt.columns.get_level_values(1),
                                     morph_pt.columns.get_level_values(3))]
    labs = [l.replace("_none", "") for l in labs]

    output_normalized_file = module.args["wide_normalized_output_file"]
    morph_pt_norm_export = morph_pt_norm.copy()
    morph_pt_norm_export.columns = labs
    morph_pt_norm_export = morph_pt_norm_export[sorted(labs, key=_natural_sort_key)]
    morph_pt_norm_export.to_csv(output_normalized_file)

    output_raw_file = module.args["wide_unnormalized_output_file"]
    morph_pt_raw_export = morph_pt.copy()
    morph_pt_raw_export.columns = labs
    morph_pt_raw_export = morph_pt_raw_export[sorted(labs, key=_natural_sort_key)]
    morph_pt_raw_export.to_csv(output_raw_file)


if __name__ == "__main__":
    main()
