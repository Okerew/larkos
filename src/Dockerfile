# Use Debian as the base image
FROM --platform=linux/amd64 debian:latest

RUN apt-get update && apt-get install -y \
    gnuplot \
    libjson-c-dev \
    build-essential \
    python3-pip \
    pybind11-dev \
    && rm -rf /var/lib/apt/lists/*


# Set up a non-root user (optional, but avoids root execution)
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    mkdir -p /app && \
    chown -R $USERNAME:$USERNAME /app

# Set working directory
WORKDIR /app

# Switch to the non-root user
USER $USERNAME

COPY --chown=$USERNAME:$USERNAME neural_web.cpp ./

RUN gcc -shared -fPIC -o neural_web.so neural_web.c -ljson-c

# Set the default command to run the program
CMD ["./neural_web"]
