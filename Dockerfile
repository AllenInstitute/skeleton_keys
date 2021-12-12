FROM continuumio/miniconda3:4.10.3
RUN conda install -y -c conda-forge rtree fenics python==3.8 jupyter mshr
RUN pip install git+https://github.com/AllenInstitute/neuron_morphology@install_improvements
RUN pip install git+https://github.com/fcollman/AllenSDK.git
ARG GITHUB_TOKEN
WORKDIR /usr/local/src
RUN git clone https://${GITHUB_TOKEN}@github.com/AllenInstitute/ccf_streamlines.git &&\
    pip install ./ccf_streamlines &&\
    rm -rf /usr/local/src/ccf_streamlines
WORKDIR /usr/local/src/skeleton_keys
COPY setup.cfg  /usr/local/src/skeleton_keys
RUN python3 -c "import configparser; c = configparser.ConfigParser(); c.read('setup.cfg'); print(c['options']['install_requires'])" | xargs pip install
COPY . /usr/local/src/skeleton_keys
RUN python setup.py install
ENV HDF5_DISABLE_VERSION_CHECK=1
