###############################################################################
# AMD BASE IMAGE
FROM rocm/dev-ubuntu-22.04:6.3.1-complete AS amd
ARG PYTORCH_ROCM_ARCH=gfx90a;gfx942
ENV PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}
ARG PYTHON_VERSION=3.12
ARG HIPBLASLT_BRANCH="4d40e36"
ARG HIPBLAS_COMMON_BRANCH="7c1566b"
ARG PYTORCH_BRANCH="3ac5a49"
ARG PYTORCH_VISION_BRANCH="v0.19.1"
ARG PYTORCH_REPO="https://github.com/pytorch/pytorch.git"
ARG PYTORCH_VISION_REPO="https://github.com/pytorch/vision.git"
ARG FA_BRANCH="b7d29fb"
ARG FA_REPO="https://github.com/ROCm/flash-attention.git"

ARG TRITON_BRANCH="e5be006"
ARG TRITON_REPO="https://github.com/triton-lang/triton.git"
ARG MAX_JOBS=$(nproc)
ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir -p /app
WORKDIR /app

# Install Python and other dependencies
RUN apt-get update -y \
    && apt-get install -y software-properties-common git curl sudo vim less \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
       python${PYTHON_VERSION}-lib2to3 python-is-python3  \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version \
    && pip install uv \
    && pip install -U packaging cmake ninja wheel setuptools pybind11 Cython \
    && apt-get update && apt-get install -y git && apt-get install -y wget && apt-get install -y cmake  && apt-get install -y python3.10-venv \
    && apt-get update && apt-get install -y openmpi-bin libopenmpi-dev 

RUN mkdir -p /app/install

RUN git clone https://github.com/ROCm/hipBLAS-common.git \
    && cd hipBLAS-common \
    && git checkout ${HIPBLAS_COMMON_BRANCH} \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make package \
    && dpkg -i ./*.deb


# Install CMake 3.26.2 temporarily
RUN mkdir -p /opt/cmake && \
    wget -O - https://github.com/Kitware/CMake/releases/download/v3.26.2/cmake-3.26.2-linux-x86_64.tar.gz | \
    tar -xvzf - --strip-components=1 -C /opt/cmake && \
    ln -sf /opt/cmake/bin/cmake /usr/local/bin/cmake

RUN git clone https://github.com/ROCm/hipBLASLt \
    && cd hipBLASLt \
    && git checkout ${HIPBLASLT_BRANCH} \
    && ./install.sh -d --architecture ${PYTORCH_ROCM_ARCH} \
    && cd build/release \
    && make package -j${MAX_JOBS} \
    && echo "Searching for build and .deb files:" \
    && find . -type d -name "build" \
    && find . -name "*.deb"

RUN cp /app/hipBLASLt/build/release/*.deb /app/hipBLAS-common/build/*.deb /app/install

# Uninstall CMake
RUN rm -rf /opt/cmake && rm -f /usr/local/bin/cmake
# Reinstall latest Cmake
RUN apt-get update && apt-get install -y cmake

RUN git clone ${TRITON_REPO}
RUN cd triton \
    && git checkout ${TRITON_BRANCH} \
    && cd python \
    && python3 setup.py bdist_wheel --dist-dir=dist && \
    pip install dist/*.whl
RUN cp /app/triton/python/dist/*.whl /app/install

# Set working directory
WORKDIR /pytorch

ENV ROCM_PATH=/opt/rocm
ENV PATH=$ROCM_PATH/bin:$ROCM_PATH/hip/bin:$PATH
ENV LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
ENV USE_ROCM=1
ENV USE_MPI=1
# Set environment variables
ENV MPI_HOME=/opt/ompi-rocm
ENV PATH=$MPI_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH
ENV CMAKE_PREFIX_PATH=$MPI_HOME:$CMAKE_PREFIX_PATH
ENV MPI_INCLUDE_DIR=$MPI_HOME/include
ENV MPI_LIBRARY=$MPI_HOME/lib/libmpi.so

# Clone and build UCX
RUN apt-get update && apt-get install -y autoconf
RUN git clone https://github.com/openucx/ucx.git -b v1.15.x && \
    cd ucx && \
    ./autogen.sh && \
    ./configure --prefix=/opt/ucx-rocm \
      --with-rocm=$ROCM_PATH \
      --enable-mt && \
    make -j$(nproc) && make install

# Download and build Slurm with system PMIx support
RUN apt-get update && apt-get install -y libpam0g-dev libnuma-dev libhwloc-dev libpmix-dev libpmix2 libmunge-dev munge
ARG SLURM_VERSION=23.11.1
RUN wget https://download.schedmd.com/slurm/slurm-${SLURM_VERSION}.tar.bz2 && \
    tar -xjf slurm-${SLURM_VERSION}.tar.bz2 && \
    cd slurm-${SLURM_VERSION} && \
    PKG_CONFIG_PATH="/usr/lib/x86_64-linux-gnu/pkgconfig" \
    CPPFLAGS="-I/usr/lib/x86_64-linux-gnu/pmix2/include" \
    LDFLAGS="-L/lib/x86_64-linux-gnu" \
    ./configure \
        --prefix=/usr/local \
        --sysconfdir=/etc/slurm \
        --with-pmix \
        --enable-pam \
        --with-pam_dir=/lib/x86_64-linux-gnu/security && \
    make -j$(nproc) && \
    make install && \
    mkdir -p /var/log/slurm && \
    mkdir -p /var/spool/slurmd && \
    chown slurm:slurm /var/spool/slurmd 2>/dev/null || true && \
    ldconfig

# Build ROCM-Aware Open MPI
RUN cd / && \
      wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.7.tar.gz && \
      tar -xvf openmpi-5.0.7.tar.gz && \
      cd openmpi-5.0.7 && \
      ./configure --prefix=/opt/ompi-rocm \
        --with-ucx=/opt/ucx-rocm \
        --with-rocm=$ROCM_PATH \
        --with-slurm \
	--with-pmix-libdir=/lib/x86_64-linux-gnu \
	--with-pmix-headers=/usr/lib/x86_64-linux-gnu/pmix2/include \
        --enable-orterun-prefix-by-default && \
      make -j$(nproc) && make install

# Verify Slurm support was built
RUN [ ! -z "$(/opt/ompi-rocm/bin/ompi_info | grep -i slurm)" ] 
 
# Set OpenMPI env variables
ENV OMPI_MCA_pml=ucx
ENV OMPI_MCA_osc=ucx
ENV OMPI_MCA_coll_ucc_enable=1
ENV OMPI_MCA_coll_ucc_priority=100
ENV UCX_TLS=sm,self,rocm


# Clone and build PyTorch
RUN git clone ${PYTORCH_REPO} pytorch && \
    cd pytorch && \
    git checkout ${PYTORCH_BRANCH} && \
    pip install -r requirements.txt && \
    git submodule update --init --recursive && \
    python3 tools/amd_build/build_amd.py && \
    python3 setup.py bdist_wheel --dist-dir=dist && \
    pip install dist/*.whl

# Clone and build torchvision
RUN git clone ${PYTORCH_VISION_REPO} vision && \
    cd vision && \
    git checkout ${PYTORCH_VISION_BRANCH} && \
    python3 setup.py bdist_wheel --dist-dir=dist && \
    pip install dist/*.whl

# Clone and build flash-attention
RUN git clone ${FA_REPO} flash-attention && \
    cd flash-attention && \
    git checkout ${FA_BRANCH} && \
    git submodule update --init && \
    MAX_JOBS=${MAX_JOBS} GPU_ARCHS=${PYTORCH_ROCM_ARCH} python3 setup.py bdist_wheel --dist-dir=dist

# Copy all wheel files to installation directory
RUN cp /pytorch/pytorch/dist/*.whl /app/install && \
    cp /pytorch/vision/dist/*.whl /app/install && \
    cp /pytorch/flash-attention/dist/*.whl /app/install

# Use test command to check if the directory exists
RUN if [ -d "/install" ]; then \
    dpkg -i /install/*.deb \
    && sed -i 's/, hipblaslt-dev \(.*\), hipcub-dev/, hipcub-dev/g' /var/lib/dpkg/status \
    && sed -i 's/, hipblaslt \(.*\), hipfft/, hipfft/g' /var/lib/dpkg/status \
    && pip install /install/*.whl; \
    fi

RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

RUN pip install /app/install/*.whl

#############################################################################
