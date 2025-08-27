###############################################################################
# AMD BASE IMAGE
FROM rocm/dev-ubuntu-22.04:6.4.1-complete AS amd
ARG PYTORCH_ROCM_ARCH=gfx90a;gfx942
ENV PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}
ARG PYTHON_VERSION=3.12
ARG HIPBLASLT_BRANCH="aa0bda7b"
ARG HIPBLAS_COMMON_BRANCH="7c1566b"
ARG PYTORCH_BRANCH="295f2ed4"
ARG PYTORCH_VISION_BRANCH="v0.21.0"
ARG PYTORCH_REPO="https://github.com/pytorch/pytorch.git"
ARG PYTORCH_VISION_REPO="https://github.com/pytorch/vision.git"
ARG FA_BRANCH="1a7f4dfa"
ARG FA_REPO="https://github.com/ROCm/flash-attention.git"
ARG AITER_BRANCH="6b92d30d"
ARG AITER_REPO="https://github.com/ROCm/aiter.git"

ARG SLURM_VERSION=23.11.1
ARG PMIX_VERSION=5.0.8

ARG TRITON_BRANCH="e5be006"
ARG TRITON_REPO="https://github.com/triton-lang/triton.git"
ARG MAX_JOBS=$(nproc)
ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir -p /app
WORKDIR /app

RUN apt-get update -y \
    && apt-get install -y software-properties-common git curl sudo vim less libgfortran5 \
    && for i in 1 2 3; do \
        add-apt-repository -y ppa:deadsnakes/ppa && break || \
        { echo "Attempt $i failed, retrying in 5s..."; sleep 5; }; \
    done \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
       python${PYTHON_VERSION}-lib2to3 python-is-python3  \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version \
    && pip install uv \
    && pip install -U packaging 'cmake<4' ninja wheel setuptools pybind11 Cython

RUN mkdir -p /app/install

RUN git clone https://github.com/ROCm/hipBLAS-common.git \
    && cd hipBLAS-common \
    && git checkout ${HIPBLAS_COMMON_BRANCH} \
    && mkdir build \
    && cd build \
    && cmake .. \
    && sed -i 's/set(CPACK_DEBIAN_PACKAGE_RELEASE "[a-f0-9]\{7\}")/set(CPACK_DEBIAN_PACKAGE_RELEASE "83~22.04")/' CPackConfig.cmake \
    && sed -i 's/set(CPACK_PACKAGE_VERSION "1.0.0")/set(CPACK_PACKAGE_VERSION "1.0.0.60401")/' CPackConfig.cmake \
    && make package \
    && dpkg -i ./*.deb


RUN git clone https://github.com/ROCm/hipBLASLt
RUN cd hipBLASLt \
    && git checkout ${HIPBLASLT_BRANCH} \
    && apt-get install -y llvm-dev \
    && ./install.sh -dc --architecture ${PYTORCH_ROCM_ARCH} ${LEGACY_HIPBLASLT_OPTION} \
    && cd build/release \
    && make package
RUN mkdir -p /app/install && cp /app/hipBLASLt/build/release/*.deb /app/hipBLAS-common/build/*.deb /app/install

RUN cp /app/hipBLASLt/build/release/*.deb /app/hipBLAS-common/build/*.deb /app/install

ENV MAX_JOBS=64
RUN git clone ${TRITON_REPO}
RUN cd triton \
    && git checkout ${TRITON_BRANCH} \
    && cd python \
    && python3 setup.py bdist_wheel --dist-dir=dist && \
    pip install dist/*.whl
RUN cp /app/triton/python/dist/*.whl /app/install

RUN cd /opt/rocm/share/amd_smi \
    && pip wheel . --wheel-dir=dist
RUN mkdir -p /app/install && cp /opt/rocm/share/amd_smi/dist/*.whl /app/install

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
RUN apt-get update -y && apt-get install -y autoconf libtool
RUN git clone https://github.com/openucx/ucx.git -b v1.15.x && \
    cd ucx && \
    ./autogen.sh && \
    ./configure --prefix=/opt/ucx-rocm \
      --with-rocm=$ROCM_PATH \
      --enable-mt && \
    make -j$(nproc) && make install

# Download and build PMIx
RUN apt-get update -y && apt-get install -y libevent-dev libhwloc-dev 
ENV PMIX_DIR=/opt/pmix
RUN wget https://github.com/openpmix/openpmix/releases/download/v${PMIX_VERSION}/pmix-${PMIX_VERSION}.tar.gz && \
    tar -xzf pmix-${PMIX_VERSION}.tar.gz && \
    cd pmix-${PMIX_VERSION} && \
    ./configure --prefix=${PMIX_DIR} \
                --enable-shared \
                --disable-static && \
    make -j$(nproc) && \
    make install

ENV PATH="${PMIX_DIR}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${PMIX_DIR}/lib:${LD_LIBRARY_PATH}"
ENV PKG_CONFIG_PATH="${PMIX_DIR}/lib/pkgconfig:${PKG_CONFIG_PATH}"

# Download and build Slurm with system PMIx support
RUN apt-get update && apt-get install -y libpam0g-dev libnuma-dev libhwloc-dev libmunge-dev munge
RUN wget https://download.schedmd.com/slurm/slurm-${SLURM_VERSION}.tar.bz2 && \
    tar -xjf slurm-${SLURM_VERSION}.tar.bz2 && \
    cd slurm-${SLURM_VERSION} && \
    PKG_CONFIG_PATH="${PMIX_DIR}/lib/pkgconfig" \
    CPPFLAGS="-I${PMIX_DIR}/include" \
    LDFLAGS="-L${PMIX_DIR}/lib" \
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
	--with-pmix-libdir=$PMIX_DIR/lib \
	--with-pmix-headers=$PMIX_DIR/include \
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
    MAX_JOBS=$(nproc) GPU_ARCHS=${PYTORCH_ROCM_ARCH} python3 setup.py bdist_wheel --dist-dir=dist

WORKDIR /app

# Clone and build aiter
RUN git clone --recursive ${AITER_REPO}
RUN cd aiter \
    && git checkout ${AITER_BRANCH} \
    && git submodule update --init --recursive \
    && pip install -r requirements.txt
RUN pip install pyyaml && cd aiter && PREBUILD_KERNELS=1 GPU_ARCHS=gfx942 python3 setup.py bdist_wheel --dist-dir=dist && ls /app/aiter/dist/*.whl
RUN mkdir -p /app/install && cp /app/aiter/dist/*.whl /app/install

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

