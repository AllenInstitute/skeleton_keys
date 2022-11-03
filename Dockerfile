FROM condaforge/mambaforge:4.9.2-5 as conda
RUN apt-get update && \
    apt-get install -y build-essential libglu1 \
    libxcursor-dev libxft2 libxinerama1 \
    libfltk1.3-dev libfreetype6-dev \
    libgl1-mesa-dev  && \
    rm -rf /var/lib/apt/lists/*
RUN mamba install -y -c conda-forge conda-lock conda-pack
COPY conda-lock.yml .
RUN conda-lock install --name skeleton_keys conda-lock.yml
SHELL ["mamba", "run","-n","skeleton_keys","/bin/bash", "-c"]
COPY requirements.txt .
RUN mamba run -n skeleton_keys pip install -r requirements.txt
COPY . /usr/local/src/skeleton_keys
WORKDIR /usr/local/src/skeleton_keys
RUN mamba run -n skeleton_keys pip install .
# RUN conda-pack -n skeleton_keys -o /tmp/env.tar && \
#     mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
#     rm /tmp/env.tar
# FROM debian:buster AS runtime
# COPY --from=conda /venv /venv
# SHELL ["/bin/bash", "-c"]
# ENTRYPOINT source /venv/bin/activate && \
#     python --version
# WORKDIR /usr/local/src/skeleton_keys
