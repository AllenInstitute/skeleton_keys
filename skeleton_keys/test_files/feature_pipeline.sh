#!bin/bash
### This is an example of running a single pair of swc file + annotation polygons (pia,wm,soma,layers) through
### feature calculation pipeline. This involves:
### 1. Uprighting the swc so that pia is up and white-matter is down
### 2. Layer aligning the cell so that laminar profiles are standardized to an average template
### 3. Extracting soma depth from pia, and laminar distribution profile (laminar histogram)
### 4. Measuring morphological features 

# Step 0, defining input/output files
specimen_id="4493"
specimen_id_file="specimen_ids.txt"
raw_orientation_swc_path="4493.swc"
surface_and_layers_file="4493.json"
layer_aligned_output_file="LayerAlignedSWCs/4493.swc"
upright_output_file="UprightSWCs/4493.swc"
histogram_output_file="AlignedLaminarHistogram.csv"
soma_depths_output_file="AlignedSomaDepths.csv"
feature_output_file="MorphologyFeatures.csv"

mkdir LayerAlignedSWCs
mkdir UprightSWCs


# Step 1, Generate upright swc file
skelekeys-upright-corrected-swc \
--output_file ${upright_output_file} \
--correct_for_shrinkage False \
--correct_for_slice_angle False \
--surface_and_layers_file ${surface_and_layers_file} \
--swc_path ${raw_orientation_swc_path} \
--specimen_id ${specimen_id} 

# Step 2, Generate the layer-aligned swc file       
skelekeys-layer-aligned-swc \
--output_file ${layer_aligned_output_file} \
--correct_for_shrinkage False \
--correct_for_slice_angle False \
--surface_and_layers_file ${surface_and_layers_file} \
--swc_path ${raw_orientation_swc_path} \
--specimen_id ${specimen_id} 

# Step 3, create laminar histogram and soma depth from pia csv files
skelekeys-profiles-from-swcs \
--specimen_id_file ${specimen_id_file} \
--swc_dir LayerAlignedSWCs \
--output_hist_file ${histogram_output_file} \
--output_soma_file ${soma_depths_output_file}

# Step 4, measure morphological features
skelekeys-morph-features \
--swc_dir UprightSWCs \
--specimen_id_file ${specimen_id_file} \
--aligned_depth_profile_file ${histogram_output_file} \
--aligned_soma_file ${soma_depths_output_file} \
--analyze_axon True \
--analyze_basal_dendrite True \
--analyze_apical_dendrite False \
--analyze_basal_dendrite_depth False \
--output_file ${feature_output_file}

