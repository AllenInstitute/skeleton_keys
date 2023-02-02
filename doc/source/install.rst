Installation
============

Supported Platforms
-------------------

We recommend installing ``skeleton_keys`` on a Linux or Mac platform; Windows is
not universally supported, though several scripts still do work on that platform.
This is primarily because its ``neuron_morphology`` dependency uses the ``fenics-dolfinx``
project, which is not natively available on Windows.


Installation Instructions
-------------------------

We recommend using ``conda`` to ensure the requirements of skeleton_keys are met.
To do this, clone the repository and set up a new conda environment.

.. code:: shell

    git clone git@github.com:AllenInstitute/skeleton_keys.git
    cd skeleton_keys
    conda create -n ENV_NAME -f environment.yml
    conda activate ENV_NAME

Then use ``pip`` to install ``skeleton_keys`` from the cloned repository.

.. code:: shell

    pip install .


