# Use CUDA runtime with cuDNN for PyTorch 1.12.1 + CUDA 11.3
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH

# Install minimal dependencies & Miniconda
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget bzip2 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/conda.sh \
    && bash /tmp/conda.sh -b -p $CONDA_DIR \
    && rm /tmp/conda.sh \
    && conda clean -afy

# Create shepherd env
COPY shepherd_env.yml /workspace/shepherd_env.yml
WORKDIR /workspace
RUN conda env create -f shepherd_env.yml \
    && conda clean -afy

# Activate shepherd env
SHELL ["conda", "run", "-n", "shepherd", "/bin/bash", "-c"]

CMD ["bash"]