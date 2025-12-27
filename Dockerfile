# GPU Development Dockerfile
# Provides NVIDIA JAX with GPU support, Java, SBT, and Python packages

FROM nvcr.io/nvidia/jax:24.04-py3

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Install Java and SBT
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    ca-certificates \
    gnupg \
    apt-transport-https \
    openjdk-11-jdk \
    && rm -rf /var/lib/apt/lists/*

# Install SBT
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823 | \
    gpg --dearmor -o /etc/apt/keyrings/sbt.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/sbt.gpg] https://repo.scala-sbt.org/scalasbt/debian all main" | \
    tee /etc/apt/sources.list.d/sbt.list && \
    apt-get update && \
    apt-get install -y sbt && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Install Python packages (JAX GPU already included in base image)
RUN pip install --upgrade \
    matplotlib \
    pandas \
    scikit-learn \
    jupyter \
    einops

# Copy project files
COPY . /workspace/

# Set Python path
ENV PYTHONPATH=/workspace/src/python

# Expose ports
EXPOSE 8888 5000

CMD ["/bin/bash"]
