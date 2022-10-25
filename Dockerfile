FROM continuumio/miniconda3:4.10.3
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
RUN conda config --set channel_priority strict 
RUN conda install -y -c conda-forge rtree==0.9.7 fenics==2019.1.0 python==3.8 hdf5==1.10.6 h5py==2.10.0 Jinja2==2.11.3
RUN pip install git+https://github.com/AllenInstitute/AllenSDK.git
RUN pip install git+https://github.com/AllenInstitute/neuron_morphology@science_staging
WORKDIR /usr/local/src
RUN git clone https://github.com/AllenInstitute/ccf_streamlines.git &&\
    pip install ./ccf_streamlines &&\
    rm -rf /usr/local/src/ccf_streamlines
WORKDIR /usr/local/src/skeleton_keys
COPY setup.cfg  /usr/local/src/skeleton_keys
RUN python3 -c "import configparser; c = configparser.ConfigParser(); c.read('setup.cfg'); print(c['options']['install_requires'])" | xargs pip install
COPY . /usr/local/src/skeleton_keys
RUN python setup.py install
