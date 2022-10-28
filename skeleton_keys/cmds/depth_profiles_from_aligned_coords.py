import os
import json
import argschema as ags
import numpy as np
import pandas as pd
from skeleton_keys.io import load_default_layer_template


class ProfilesFromAlignedCoordsParameters(ags.ArgSchema):
    coordinate_file = ags.fields.InputFile(
        description="File with specimen IDs on each line"
    )
    depth_label = ags.fields.String(
        default="y",
        description="label of column with aligned depth values",
    )
    index_label = ags.fields.String(
        default=None,
        allow_none=True,
        description="label of column to split values across rows"
    )
    hist_split_label = ags.fields.String(
        default=None,
        allow_none=True,
        description="label of column to split values across multiple sets of columns"
    )
    layer_depths_file = ags.fields.InputFile(
        description="JSON file with layer depths; used to establish bins for profile histogram",
        default=None, allow_none=True)
    bin_size = ags.fields.Float(description="bin size, in microns", default=5.0)
    below_wm = ags.fields.Float(description="extent below white matter to include, in microns", default=200.0)
    output_hist_file = ags.fields.OutputFile(
        description="output CSV file for depth profiles",
        default="aligned_histograms_corrected.csv")




def main(args):
    # Load the coordinates
    coordinate_file = args['coordinate_file']
    coord_df = pd.read_csv(coordinate_file)


    # Load the layer info
    layer_depths_file = args['layer_depths_file']
    if layer_depths_file:
        with open(layer_depths_file, "r") as f:
            avg_layer_depths = json.load(f)
    else:
        avg_layer_depths = load_default_layer_template()

    # Set up bins for profile histograms
    bin_size = args['bin_size']
    below_wm = args['below_wm']
    bottom_edge = avg_layer_depths["wm"] + below_wm
    bins = np.arange(0, bottom_edge, bin_size)

    record_list = []

    depth_label = args['depth_label']
    index_group_label = args['index_label']
    col_group_label = args['hist_split_label']

    if index_group_label is None:
        if col_group_label is None:
            # Calculating a single histogram for all values
            col_prefix = "bin"
            names = [f"{col_prefix}_{i}" for i in range(len(bins) - 1)]
            y_values = -coord_df[depth_label].values
            hist_values, _ = np.histogram(y_values, bins=bins)
            out_df = pd.DataFrame(hist_values, index=names).T
        else:
            names = []
            hist_list = []
            for col_ind, col_group in coord_df.groupby(col_group_label):
                names += [f"{col_ind}_{i}" for i in range(len(bins) - 1)]
                y_values = -col_group[depth_label].values
                hist_values, _ = np.histogram(y_values, bins=bins)
                hist_list.append(hist_values)
            out_df = pd.DataFrame(np.hstack(hist_list), index=names).T

    else:
        if col_group_label is None:
            col_prefix = "bin"
            names = [f"{col_prefix}_{i}" for i in range(len(bins) - 1)]
            group_dict = {}
            for row_ind, row_group in coord_df.groupby(index_group_label):
                y_values = -row_group[depth_label].values
                hist_values, _ = np.histogram(y_values, bins=bins)
                group_dict[row_ind] = hist_values
            out_df = pd.DataFrame(group_dict, index=names).T.rename_axis(index_group_label).reset_index()
        else:
            group_dict = {}
            all_col_names_in_order = []
            for cn, _ in coord_df.groupby(col_group_label):
                all_col_names_in_order += [f"{cn}_{i}" for i in range(len(bins) - 1)]

            data_for_df = {}
            for group_names, group_df in coord_df.groupby([index_group_label, col_group_label]):
                ind_name = group_names[0]
                if ind_name not in data_for_df:
                    data_for_df[ind_name] = {}
                col_prefix = group_names[1]

                y_values = -group_df[depth_label].values
                hist_values, _ = np.histogram(y_values, bins=bins)
                names = [f"{col_prefix}_{i}" for i in range(len(hist_values))]
                data_for_df[ind_name].update(dict(zip(names, hist_values)))

            out_df = pd.DataFrame(data_for_df).fillna(0).astype(int).T
            out_df = out_df[all_col_names_in_order]
            out_df = out_df.rename_axis(index_group_label).reset_index()

    out_df.to_csv(args['output_hist_file'], index=False)


def console_script():
    module = ags.ArgSchemaParser(schema_type=ProfilesFromAlignedCoordsParameters)
    main(module.args)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=ProfilesFromAlignedCoordsParameters)
    main(module.args)
