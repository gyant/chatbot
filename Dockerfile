# Use CUDA base image
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS build

# Install Python and build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    build-essential \
    cmake \
    git \
    ninja-build \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for CUDA
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda

# Set build args for llama-cpp-python
ENV CMAKE_ARGS="-DGGML_CUDA=ON"
ENV FORCE_CMAKE=1

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
RUN CUDACXX=/usr/local/cuda/bin/nvcc pip install llama-cpp-python --no-cache-dir
RUN rm /usr/local/cuda/lib64/stubs/libcuda.so.1


COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Runtime stage
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libgomp1 \
    vim \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

WORKDIR /app

# Install Python requirements
COPY main.py .
COPY config.yaml.template .

CMD ["python3", "main.py"]
