User's Guide
============

The ``skeleton_keys`` package supports the skeletal analysis of neuron morphologies.
It is used to perform a series of analysis steps in a consistent and scaleable
manner. It can be used to align a series of morphologies from the isocortex to a
set of reference layer depths/thicknesses, create layer-aligned depth profile
histograms, and calculate morphological features using the `neuron_morphology package <https://neuron-morphology.readthedocs.io/en/latest/>`_.

The main inputs for ``skeleton_keys`` are neuron morphologies in the form of
SWC files, layer drawings for the cells of interest, and reference information
such as a reference set of layer depths/thicknesses. The package contains
several command line scripts that are typically used in sequence to process a
set of morphologies and end up with a comparable set of morphological features
for the entire set.


Aligning a morphology to a reference set of layers
--------------------------------------------------

The depths and thicknesses of cortical layers can vary from location to location
across the isocortex. However, some cells send processes into specific layers
regardless of their location. Analyses that use the cells only at their original
size may blur precise distinctions that follow layer boundaries; therefore,
several features calculated by this package rely on first aligning morphologies
to a consistent set of layer thicknesses/depths. These layer-aligned morphologies
can also be used for other purposes, like visualization.

Since the isocortex is a curved structure, the depth is not always straightforward
to calculate. The `neuron_morphology package`_ (used by ``skeleton_keys`` to
perform the layer alignemnt) takes the approach used in the
`Allen Common Coordinate Framework for the mouse brain <http://help.brain-map.org/download/attachments/2818171/Conn_Informatics_Data_Processing.pdf?version=2&modificationDate=1507057121463&api=v2>`_,
where paths traveling through isopotential surfaces between the pia and white matter
define the depth dimension of the cortex.

Therefore, to align to a common set of layers, we need information about
where the pia, white matter, and layers are with respect to the cell of interest.
At the Allen Institute for Brain Science, these are drawn on 20x images
of the morphology, using a DAPI stain to identify layers. These lines and polygons
become inputs into the ``skeleton_keys`` scripts.

These layer drawings match the orientation of the reconstructed
cell taken directly from the image (i.e., before any rotation to place pia upward
is done).

As an example, we will use an Sst+ inhibitory neuron from the `Gouwens et al. (2020) study <https://linkinghub.elsevier.com/retrieve/pii/S0092-8674(20)31254-X>`_
where we have also saved the layer drawings in the ``example_data`` directory as
a JSON file in the format expected by ``skeleton_keys``.

In its original orientation, the morphology looks like this:

.. figure:: /images/guide_example_sst_original.png
    :width: 600

    Sst neuron in original orientation. Dendrite is red, axon is blue, soma is black dot.


Our layer drawings (with a matched orientation) look like:

.. figure:: /images/guide_example_layer_drawings.png
    :width: 600

    Pia, white matter, soma outline, and layer drawings

And together (after aligning the soma locations) they look like:

.. figure:: /images/guide_example_layer_drawings_with_morph.png
    :width: 600

    Morphology and layer drawings in original orientation


**Note:** This script also has functionality to correct the morphology for shrinkage
(which can happen because the fixed tissue that is image dries out and becomes
flatter than the original) and slice angle tilt (which happens when the
cutting angle for brain slices does not match the curvature of that part of the
brain). However, these features are currently written to require access to an
internal Allen Institute database and do not yet have alternative input formats. Therefore,
we will not use those functions in this guide.

We will supply the script with the following inputs:

* *specimen_id* - an integer identifier for the cell.
    Here, it is primarily used
    to access internal database information (which we aren't doing), but in other
    scripts it is used to associate this cell with its features. Here, the specimen ID
    of our example is ``740135032``.
* *swc_path* - a file path to the SWC file in its original orientation.
    Here, our example cell's SWC file is ``Sst-IRES-Cre_Ai14-408415.03.02.02_864876723_m.swc``.
* *surface_and_layers_file* - a JSON file with the layer drawings.
    Here, our example uses the file ``740135032_surfaces_and_layers.json``.
* *layer_depths_file* - a JSON file with the set of layer depths we're aligning the cell to.
    In this case we'll use an average set of depths included as a test
    file, ``avg_layer_depths.json``.

Therefore, our command will be:

.. code:: shell

    skelekeys-layer-aligned-swc --specimen_id 740135032 \
    --swc_path Sst-IRES-Cre_Ai14-408415.03.02.02_864876723_m.swc \
    --surface_and_layers_file 740135032_surfaces_and_layers.json \
    --layer_depths_file avg_layer_depths.json \
    --correct_for_shrinkage False \
    --correct_for_slice_angle False \
    --output_file layer_aligned_740135032.swc

This creates a new SWC file (``layer_aligned_740135032.swc``) that is (1)
uprighted and (2) stretched or squished to align each of its points to the
reference set of layers.

.. figure:: /images/guide_example_sst_layeraligned.png
    :width: 600

    Layer-aligned morphology


Uprighting a morphology without layer alignment
-----------------------------------------------

If you only want to orient morphology so that pia is up and the white matter is
down, but without making any layer thickness adjustments, you can use the
command-line utility ``skelekeys-upright-corrected-swc``. It still requires
layer drawings, though, to know which direction the pia and white matter are
relative to the originally reconstructed morphology.

**Note:** This script, too, can correct the morphology for shrinkage and slice angle tilt,
but we are again skipping that since it currently can only use internally databased
information.

It takes a similar set of arguments as before (but notably without the ``--layer_depths_file``
argument).

.. code:: shell

    skelekeys-upright-corrected-swc --specimen_id 740135032 \
    --swc_path Sst-IRES-Cre_Ai14-408415.03.02.02_864876723_m.swc \
    --surface_and_layers_file 740135032_surfaces_and_layers.json \
    --correct_for_shrinkage False \
    --correct_for_slice_angle False \
    --output_file upright_only_740135032.swc

The output looks like:

.. figure:: /images/guide_example_sst_upright.png
    :width: 600

    Upright (but not layer-aligned) morphology


Calculating depth profiles
--------------------------

A relevant aspect for distinguishing morphologies is the depth profile, which is
a 1D histogram of the number of nodes across a set of depth bins, divided by compartment
type (i.e., axon, basal dendrite, apical dendrite). These can be used to calculate
reduced-dimesion representations of those profiles, determine the overlap of different
compartment types, etc.

The command line utility ``skelekeys-profiles-from-swcs`` will create a CSV file of
depth profiles from a list of layer-aligned SWC files.

The script expects the layer-aligned SWC files to be in a single directory (``--swc_dir``)
and be named as ``{specimen_id}.swc``. For this example, we have moved the layer-aligned
Sst cell's SWC file (740135032) into another directory and renamed it; we have also layer aligned
a Pvalb cell (606271263) and a Vip cell (694146546).

.. figure:: /images/guide_three_layeraligned_cells.png
    :width: 600

    Layer-aligned Sst cell, Pvalb cell, and Vip cell

To run the script for our example, we give it the following inputs:

* *specimen_id_file* - a text file with specimen IDs
    The text file has one integer ID per line. Here we're using
    ``example_specimen_ids.txt``.
* *swc_dir* - the directory of layer-aligned SWC files
    Here our directory is ``layer_aligned_swcs``
* *layer_depths_file* - a JSON file with the reference set of layer depths
    This is so that the script knows where the white matter begins (so that it
    can determine how far past to include)
* *output_hist_file* - an output CSV file path
    This CSV file contains the depth histograms - the columns are depth bins,
    and the rows are cells
* *output_soma_file* - an output CSV file path
    The script also saves the layer-aligned soma depths, which is used by
    other command line scripts in the package

Our command will then be:

.. code:: shell

    skelekeys-profiles-from-swcs --specimen_id_file example_specimen_ids.txt \
    --swc_dir layer_aligned_swcs \
    --layer_depths_file avg_layer_depths.json \
    --output_hist_file aligned_depth_profiles.csv \
    --output_soma_file aligned_soma_depths.csv

This produces the following depth profiles:

.. figure:: /images/guide_depth_profiles.png
    :width: 600

    Layer-aligned depth profiles of the three example cells


PCA on depth profiles
---------------------

A reduced dimension representation of the depth profiles can serve as useful
features for distinguishing morphologies. However, analyses like PCA will produce
loadings that vary depending on the data set. The command line utility
``skelekeys-calc-histogram-loadings`` can be used to generate a fixed loading file
from a set of morphologies that can be used to analyze other morphologies with
other scripts.

If you simply want to use the PCA loadings for a new set of cells and not
transfer those loadings to another set, you do not need to use this utility
(you can use the ``skelekeys-morph-features`` script by itself). That script
will also let you save the loadings. But this script will allow you to calculate
loadings without having to calculate all the other morphological features.

In this example, we will calculate and save the PCA loadings for the axonal
compartments. The command to do so is:

.. code:: shell

    skelekeys-calc-histogram-loadings --specimen_id_file example_specimen_ids.txt \
    --aligned_depth_profile_file aligned_depth_profiles.csv \
    --analyze_axon True \
    --save_axon_depth_profile_loadings_file axon_loadings.csv


Calculating morphological features
----------------------------------

Once we have the layer depths files (and optionally a set of pre-calculated loadings),
we can calculate a set of morphological features for each cell using the
command line utility ``skelekeys-morph-features``.
The main inputs are the set of upright (but *not* layer-aligned) SWC files and the depth
profile CSV file. We also specify which compartments we want to analyze (for example
if we have a set of excitatory neurons that don't have reconstructed local axons,
we would not want to analyze axonal compartments).

To continue our example, we will analyze the features using the following command:

.. code:: shell
    skelekeys-morph-features  --specimen_id_file example_specimen_ids.txt \
    --swc_dir upright_swcs \
    --aligned_depth_profile_file aligned_depth_profiles.csv \
    --aligned_soma_file aligned_soma_depths.csv \
    --analyze_axon True \
    --analyze_basal_dendrite True \
    --analyze_apical_dendrite False \
    --output_file example_features_long.csv


This produces a long-form feature data file.

.. csv-table:: Beginning of example long-form feature file
   :file: guide_example_features_long_excerpt.csv
   :header-rows: 1



Post-processing morphological features
--------------------------------------

The long-form file can be used for many purposes,
but it is also useful to convert the data to a wide format where the
features are the columns and the rows are cells. At the same time,
it can also be useful to normalize the different features for analyzes like
classification and clustering.

The command line utility ``skelekeys-postprocess-features`` is used to perform
these operations.

.. code:: shell

    skelekeys-postprocess-features \
    --input_files "['example_features_long.csv']" \
    --wide_normalized_output_file example_features_wide_normalized.csv \
    --wide_unnormalized_output_file example_features_wide_unnormalized.csv

Note the `syntax for passing a list of files <https://argschema.readthedocs.io/en/latest/user/intro.html#command-line-specification>`_ for the `argschema <https://argschema.readthedocs.io/en/latest/>`_
command line argument. If you passed more than one file (for example, to
normalize features calculated from two sets of cells to the same scale), you
would separate each argument with a comma (as in ``"['file_one.csv','file_two.csv']"``).

The wide form feature file output looks like:

.. csv-table:: Example wide-form feature file
   :file: guide_example_features_wide.csv
   :header-rows: 1


Working with general coordinates
--------------------------------

The ``skeleton_keys`` package was originally built around processing SWC files,
but it can also be used to align arbitrary sets of coordinates, if provided with
the appropriate layer drawings.

The command line utility ``skelekeys-layer-aligned-coords`` can take a CSV and
adjust the specified coordinates to make them layer-aligned. It takes the following
inputs:

* *coordinate_file* - a CSV with coordinates
    The coordinate columns must contain "x", "y", and "z" in their names,
    but can have prefixes and/or suffixes (see below). We'll use an example file
    named ``coord_example.csv``.
* *layer_depths_file*  a JSON file with the set of layer depths we're aligning the cell to.
    Here again we'll use an average set of depths included as a test
    file, ``avg_layer_depths.json``.
* *surface_and_layers_file* - a JSON file with the layer drawings.
    Here, our example uses the file ``coord_layer_drawings.json``.
* *coordinate_column_prefix (and/or _suffix)* - strings of common prefixes/suffixes
    This allows the coordinate columns to have names other than the default ``x``,
    ``y``, and ``z``, but in this example we do not need to use them.

Using this, we can take a starting example file (the columns ``cell_id`` and
``target_cell_type`` contain extra metadata about the coordinates):

.. csv-table:: Example coordinate file
   :file: guide_coord_example.csv
   :header-rows: 1

Use the command:

.. code:: shell

    skelekeys-layer-aligned-coords \
    --coordinate_file coord_example.csv \
    --surface_and_layers_file coord_layer_drawings.json \
    --layer_depths_file avg_layer_depths.json \
    --output_file aligned_coord_example.csv

And obtain:

.. csv-table:: Example coordinate file
   :file: guide_aligned_coord_example.csv
   :header-rows: 1

Note that the ``x`` values have also changed because we have rotated the
coordinates to an upright orientation (with pia at the top).

This aligned coordinate file can be used to generate histograms with the
``skelekeys-profiles-from-coords`` command line utility. You need to specify
which column contains the depth values (here, the column labeled ``y``) with the
``depth_label`` argument.

You can use other columns in the CSV file to split the histograms across rows using the
``--index_label`` argument, and/or you can create multiple histograms per row (as in the
compartment type histograms above) with the ``--hist_split_label`` argument.

For example:

.. code:: shell

    skelekeys-profiles-from-coords \
    --coordinate_file aligned_coord_example.csv \
    --layer_depths_file avg_layer_depths.json \
    --depth_label y \
    --index_label cell_id \
    --hist_split_label target_cell_type \
    --output_hist_file aligned_coord_hist.csv
