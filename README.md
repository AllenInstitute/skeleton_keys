# skeleton keys

This package supports the skeletal analysis of morphologies.


Installation instructions
=========================


install ccf_streamlines from source
https://github.com/AllenInstitute/ccf_streamlines
git@github.com:AllenInstitute/ccf_streamlines.git

install neuron_morphology from source
https://github.com/AllenInstitute/neuron_morphology.git
git@github.com:AllenInstitute/neuron_morphology.git

clone source
navigate to source directory 
pip install .

Scripts
=======
After installation the following console scripts will be available to run from the command line of your environment. To see detailed instructions on each script type the name of the SCRIPT_NAME --help

skelekeys-layer-aligned-swc
----------------------------
script to take an swc file, a polygon definition, a common layer cortical boundary file and produce a swc file that has been remapped to a common layer. 

depth_profiles_from_aligned_swcs
--------------------------------
script to take an directory of swc files, a common layer boundary file and return layer histogram file and a soma position file for the cells

skelekeys-morph-features
----------------------------
script to take a list of layer aligned cells, a layer histogram file, a soma position file, and a common cortical boundary file and extract morphological features from those cells in a long format.

skelekeys-postprocess-features
------------------------------
script to to take a long form set of features and post-process them to remove zeros and perform zscoring and/or uniform scaling of features.








