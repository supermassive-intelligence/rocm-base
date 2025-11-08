###############################################################################
# AMD BASE IMAGE
FROM rocm/dev-ubuntu-22.04:7.0-complete AS amd
ARG PYTORCH_ROCM_ARCH=gfx942;gfx950
ENV PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}
ARG PYTHON_VERSION=3.12
ARG HIPBLASLT_BRANCH="634088c98d"
ARG HIPBLASLT_REPO=https://github.com/ROCm/rocm-libraries.git
ARG HIPBLAS_COMMON_BRANCH="7c1566b"
ARG PYTORCH_BRANCH="1c57644d"
ARG PYTORCH_VISION_BRANCH="v0.23.0"
ARG PYTORCH_REPO="https://github.com/ROCm/pytorch.git"
ARG PYTORCH_VISION_REPO="https://github.com/pytorch/vision.git"
ARG FA_BRANCH="0e60e394"
ARG FA_REPO="https://github.com/ROCm/flash-attention.git"
ARG AITER_BRANCH="9716b1b8"
ARG AITER_REPO="https://github.com/ROCm/aiter.git"

ARG SLURM_VERSION=23.11.1
ARG PMIX_VERSION=5.0.8

ARG TRITON_BRANCH="57c693b6"
ARG TRITON_REPO="https://github.com/ROCm/triton.git"
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
    && sed -i 's/set(CPACK_DEBIAN_PACKAGE_RELEASE "[a-f0-9]\{7\}")/set(CPACK_DEBIAN_PACKAGE_RELEASE "38~22.04")/' CPackConfig.cmake \
    && sed -i 's/set(CPACK_PACKAGE_VERSION "1.0.0")/set(CPACK_PACKAGE_VERSION "1.2.0.70000")/' CPackConfig.cmake \
    && make package \
    && dpkg -i ./*.deb

RUN apt-get --purge remove -y hipblaslt hipblaslt-dev \
    && rm -fr /opt/rocm/include/hipblaslt \
    && rm -fr /opt/rocm/lib/libhipblaslt* \
    && rm -rf /opt/rocm-7.0.0/lib/hipblaslt

# Clone and build hipBLASLt from source
RUN git clone ${HIPBLASLT_REPO} \
    && cd rocm-libraries/projects/hipblaslt \
    && git checkout ${HIPBLASLT_BRANCH} \
    && git cherry-pick e0deaed2fa081e904b19e09d73257c1d4a817e99 --strategy-option theirs || true \
    && git cherry-pick 5e2e486d6568b1b8b9f1656edae75a3fe45288a1 --strategy-option theirs || true \
    && python3 -m pip install -r tensilelite/requirements.txt \
    && MAX_JOBS=${MAX_JOBS} ./install.sh -idc --architecture "${PYTORCH_ROCM_ARCH}" \
    && cd ../../../ \
    && rm -fr rocm-libraries

RUN git clone ${TRITON_REPO}
RUN cd triton \
    && git checkout ${TRITON_BRANCH} \
    && if [ ! -f setup.py ]; then cd python; fi \
    && MAX_JOBS=32 python3 setup.py bdist_wheel --dist-dir=dist \
    && mkdir -p /app/install && cp dist/*.whl /app/install
RUN if [ -d triton/python/triton_kernels ]; then pip install build && cd triton/python/triton_kernels \
    && python3 -m build --wheel && cp dist/*.whl /app/install; fi

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
RUN apt-get update -y && apt-get install -y autoconf libtool libibverbs-dev librdmacm-dev rdma-core ibverbs-providers
RUN git clone https://github.com/openucx/ucx.git -b v1.15.x && \
    cd ucx && \
    ./autogen.sh && \
    ./configure --prefix=/opt/ucx-rocm \
      --with-rocm=$ROCM_PATH \
      --with-verbs \
      --with-rdmacm \
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
ENV UCX_TLS=all
ENV LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH}"

# Clone and build PyTorch
RUN apt-get update && apt-get install -y rocblas-dev hipblas-dev miopen-hip hipsolver-dev
RUN git clone ${PYTORCH_REPO} pytorch && \
    cd pytorch && \
    git checkout ${PYTORCH_BRANCH} && \
    pip install -r requirements.txt && \
    git submodule update --init --recursive && \
    python3 tools/amd_build/build_amd.py && \
    MAX_JOBS=$(nproc) CMAKE_PREFIX_PATH=$(python3 -c 'import sys; print(sys.prefix)') python3 setup.py bdist_wheel --dist-dir=dist && \
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
RUN pip install pyyaml && cd aiter && PREBUILD_KERNELS=1 GPU_ARCHS=${PYTORCH_ROCM_ARCH} MAX_JOBS=192 python3 setup.py bdist_wheel --dist-dir=dist && ls /app/aiter/dist/*.whl
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

