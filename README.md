# skeleton keys

This package supports the skeletal analysis of morphologies.

We recommend installing this on Linux or Mac via conda. We aren't generally currently supporting Windows, though there are some scripts which may work and be useful on that platform. This is because we are utilized fenics-dolphix which is not available on Windows. 

Installation instructions
=========================

clone the repo, setup a conda environment with the proper requirements and activate it

    git clone git@github.com:AllenInstitute/skeleton_keys.git
    cd skeleton_keys
    conda env create -f environment.yml
    conda activate skeleton_keys_env

pip install skeleton keys

    pip install .

Internal Allen Institute Use
============================
In order to download morphologies from the internal LIMS system, you must set certain environment variables to connect properly.  These include

        LIMS_HOST
        LIMS_DBNAME
        LIMS_USER
        LIMS_PASSWORD

Contact the technology team if you need to get credential details to access LIMS.

Scripts
=======
After installation the following console scripts will be available to run from the command line of your environment. To see detailed instructions on each script type the name of the SCRIPT_NAME --help

skelekeys-layer-aligned-swc
----------------------------
script to take a swc file, a polygon definition (representing cortical layers and pia/wm surfaces), a common layer cortical boundary file and produce a swc file that has been remapped to common layer boundaries.

skelekeys-profiles-from-swcs
--------------------------------
script to take a directory of layer-aligned swc files, a common layer boundary file and return layer histogram file and a soma depth file for the cells

skelekeys-morph-features
----------------------------
script to take a list of upright cells, a layer histogram file, and a soma depth file, and extract morphological features from those cells in a long format.

This script can be run without upright cells, and/or a layer histogram file, and/or a soma depth file. Just pass in the swc files that you would like to quantify and leave the histogram and soma depth file arguments as the default (None).

skelekeys-postprocess-features
------------------------------
script to take a long form set of features and post-process them to remove zeros and perform zscoring and/or uniform scaling of features.

Statement of Support
====================
This code is an important part of the internal Allen Institute code base and we are actively using and maintaining it. Issues are encouraged, but because this tool is so central to our mission pull requests might not be accepted if they conflict with our existing plans.






