FROM condaforge/mambaforge:4.9.2-5 as conda
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
COPY conda-lock.yml .
RUN mamba install -y -c conda-forge conda-lock conda-pack
RUN conda-lock install --name skeleton_keys conda-lock.yml
SHELL ["mamba", "run","-n","skeleton_keys","/bin/bash", "-c"]
RUN mamba run -n skeleton_keys python --version
RUN pip install git+https://github.com/AllenInstitute/neuron_morphology@science_staging
RUN pip install git+https://github.com/AllenInstitute/ccf_streamlines.git
RUN conda-pack -n skeleton_keys -o /tmp/env.tar && \
    mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
    rm /tmp/env.tar
COPY requirements.txt .
RUN mkdir skeleton_keys && source /venv/bin/activate && pip install -r requirements.txt
#RUN source /venv/bin/activate && python3 -c "import configparser; c = configparser.ConfigParser(); c.read('setup.cfg'); print(c['options']['install_requires'])" | xargs pip install

COPY . /usr/local/src/skeleton_keys
WORKDIR /usr/local/src/skeleton_keys
RUN source /venv/bin/activate && python setup.py install

FROM debian:buster AS runtime
COPY --from=conda /venv /venv
SHELL ["/bin/bash", "-c"]
ENTRYPOINT source /venv/bin/activate && \
    python --version
WORKDIR /usr/local/src/skeleton_keys

