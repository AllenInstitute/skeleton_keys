FROM continuumio/miniconda3:4.10.3
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
RUN conda install -y -c conda-forge rtree fenics python==3.8 jupyter mshr hdf5
RUN pip install git+https://github.com/fcollman/AllenSDK.git
RUN pip install git+https://github.com/AllenInstitute/neuron_morphology@install_improvements
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
