import os

import argschema as ags
import cloudfiles
import numpy as np
import pandas as pd

from skeleton_keys import cloudfields
from skeleton_keys.io import (fix_local_cloudpath, load_default_layer_template,
                              load_swc_as_dataframe, read_bytes,
                              read_json_file, write_dataframe_to_csv)


class ProfilesFromAlignedSwcsParameters(ags.ArgSchema):
    specimen_id_file = cloudfields.InputFile(
        description="File with specimen IDs on each line"
    )
    swc_dir = cloudfields.InputDir(description="Directory with layer-aligned SWC files")
    layer_depths_file = cloudfields.InputFile(
        description="JSON file with layer depths; used to establish bins for profile histogram",
        default=None,
        allow_none=True,
    )
    bin_size = ags.fields.Float(description="bin size, in microns", default=5.0)
    below_wm = ags.fields.Float(
        description="extent below white matter to include, in microns", default=200.0
    )
    output_hist_file = cloudfields.OutputFile(
        description="output CSV file for depth profiles",
        default="aligned_histograms_corrected.csv",
    )
    output_soma_file = cloudfields.OutputFile(
        description="output CSV file for soma depths",
        default="aligned_soma_depths_corrected.csv",
    )


def main(module=None):
    if module == None:
        module = ags.ArgSchemaParser(schema_type=ProfilesFromAlignedSwcsParameters)

    # Load the specimen IDs
    specimen_id_file = module.args["specimen_id_file"]
    specimen_ids = np.loadtxt(read_bytes(specimen_id_file), dtype=int, ndmin=1)

    # Load the layer info
    layer_depths_file = module.args["layer_depths_file"]
    if layer_depths_file:
        avg_layer_depths = read_json_file(layer_depths_file)
    else:
        avg_layer_depths = load_default_layer_template()

    # Get directory with layer-aligned SWCs
    swc_dir = module.args["swc_dir"]

    # Set up bins for profile histograms
    bin_size = module.args["bin_size"]
    below_wm = module.args["below_wm"]
    bottom_edge = avg_layer_depths["wm"] + below_wm
    bins = np.arange(0, bottom_edge, bin_size)

    hist_record_list = []
    soma_record_list = []
    swc_dir = fix_local_cloudpath(swc_dir)
    cf = cloudfiles.CloudFiles(swc_dir)
    files_exist = cf.exists([f"{spec_id}.swc" for spec_id in specimen_ids])
    for spec_id in specimen_ids:
        try:
            # Load individual morphology
            swc_file = f"{spec_id}.swc"
            if not files_exist[swc_file]:
                print(f"No SWC file found in {swc_dir} for {spec_id}")
                continue
            swc_file = os.path.join(swc_dir, swc_file)
            morph_df = load_swc_as_dataframe(swc_file)

            # Create the layer-aligned histogram
            result = {}
            node_types = [2, 3, 4]
            for nt in node_types:
                subset_df = morph_df.loc[morph_df["type"] == nt, :]
                y_values = -subset_df["y"].values
                hist_values, _ = np.histogram(y_values, bins=bins)
                result[nt] = hist_values

            flat_result_dict = {"specimen_id": spec_id}
            for nt in result:
                for i in range(len(result[nt])):
                    k = f"{nt}_{i}"
                    flat_result_dict[k] = result[nt][i]
            # Pull out the soma depths
            soma_y = morph_df.loc[morph_df["type"] == 1, "y"].values[0]
            soma_record_list.append(
                {
                    "specimen_id": spec_id,
                    "soma_distance_from_pia": -soma_y,
                }
            )
            hist_record_list.append(flat_result_dict)
        except:
            print(f"Error processing {spec_id}")

    # Save results
    output_hist_file = module.args["output_hist_file"]
    output_soma_file = module.args["output_soma_file"]
    write_dataframe_to_csv(
        pd.DataFrame.from_records(hist_record_list)
        .fillna(0.0)
        .set_index("specimen_id"),
        output_hist_file,
    )
    write_dataframe_to_csv(
        pd.DataFrame.from_records(soma_record_list)
        .fillna(0.0)
        .set_index("specimen_id"),
        output_soma_file,
    )


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
