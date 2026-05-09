# Use the NVIDIA CUDA base image with Debian
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Install dependencies (no sudo required in Docker)
RUN apt-get update && apt-get install -y \
    clang \
    gnuplot \
    libjson-c-dev \
    libcurl4-openssl-dev \
    build-essential \
    python3-pip \
    python3-setuptools \
    python3-numpy\
    pybind11-dev \
    python3-pybind11 \
    && rm -rf /var/lib/apt/lists/*

# Set up a non-root user (optional, but avoids root execution)
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    chown -R $USERNAME:$USERNAME /app

# Set working directory
WORKDIR /app

# Switch to the non-root user
USER $USERNAME

COPY --chown=$USERNAME:$USERNAME cuda_kernels.cu  ./

COPY --chown=$USERNAME:$USERNAME neural_kernels.cpp  ./

COPY --chown=$USERNAME:$USERNAME setup.py ./

RUN python3 setup.py sdist bdist_wheel

# Set the default command to run the program
CMD ["./neural_web"]
