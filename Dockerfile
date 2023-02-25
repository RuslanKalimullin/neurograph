# adapted from https://github.com/sinzlab/pytorch-docker/blob/master/Dockerfile
# and https://fabiorosado.dev/blog/install-conda-in-docker/
FROM nvidia/cuda:11.4.0-runtime-ubuntu18.04 as base

# Install base utilities
RUN apt-get update && \
    apt-get install -y build-essential wget vim  && \
    apt-get clean && \
    # best practice to keep the Docker image lean
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install miniconda (yes) with required python version
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Install requirements
COPY ./pyg_cpu.sh /tmp
RUN /tmp/pyg_cpu.sh

COPY ./requirements.txt /tmp
RUN python -m pip install -r /tmp/requirements.txt

# Deal with pesky Python 3 encoding issue
ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
ENV MPLLOCALFREETYPE 1

WORKDIR /app

# Export port for Jupyter Notebook
EXPOSE 8888
