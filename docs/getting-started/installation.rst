============
Installation
============

------------------------------------
Build Triton-distributed from source
------------------------------------


----------------------------------
Best practice for Nvidia backend:
----------------------------------

+++++++++++++
Requirements:
+++++++++++++
- Python >=3.11 (suggest using virtual environment)
- CUDA >=12.4
- Torch >=2.4.1
- Clang >=19

We recommend installation in `Nvidia PyTorch container <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags>`_.



Dependencies with other versions may also work well, but this is not guaranteed. If you find any problem in installing, please tell us in Issues.

++++++
Steps:
++++++

1. **Prepare docker container:**

   .. code-block:: sh

      docker run --name triton-dist --ipc=host --network=host --privileged --cap-add=SYS_ADMIN --shm-size=10g --gpus=all -itd nvcr.io/nvidia/pytorch:25.04-py3 /bin/bash
      docker exec -it triton-dist /bin/bash

2. **Clone Triton-distributed to your own path (e.g., `/workspace/Triton-distributed`)**

   .. code-block:: sh

      git clone https://github.com/ByteDance-Seed/Triton-distributed.git

3. **Update submodules**

   .. code-block:: sh

      cd /workspace/Triton-distributed
      git submodule update --init --recursive

4. **Install dependencies (optional for PyTorch container)**

   .. note:: Not needed for PyTorch container

   .. code-block:: sh

      # If you are not using PyTorch container
      pip3 install torch==2.4.1
      pip3 install cuda-python==12.4 # need to align with your nvcc version
      pip3 install ninja cmake wheel pybind11 numpy chardet pytest

5. **Prepare NVSHMEM and Clang-19.**

   See `the guide <prepare_nvshmem.md>`_ to prepare NVSHMEM.

   Clang-19 is required to build NVSHMEM bitcode library and Triton. To install Clang-19, we recommend pre-built binary:

   .. code-block:: sh

      apt update
      apt install clang-19 llvm-19 libclang-19-dev

   Also, you may install Clang-19 from source by building LLVM (see `how to build LLVM <https://llvm.org/docs/CMake.html>`_).

   .. code-block:: sh

      git clone git@github.com:llvm/llvm-project.git
      cd llvm-project
      git checkout llvmorg-19.1.0
      mkdir build
      cd build
      cmake -G Ninja ../llvm    -DLLVM_ENABLE_PROJECTS="clang;lldb;lld"    -DLLVM_BUILD_EXAMPLES=ON    -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU"    -DCMAKE_BUILD_TYPE=Release    -DLLVM_ENABLE_ASSERTIONS=ON    -DMLIR_ENABLE_BINDINGS_PYTHON=ON  -DCMAKE_BUILD_TYPE=Release
      cmake --build .

   Remember to put the built binary and library path to `PATH` and `LD_LIBRARY_PATH`.

   .. code-block:: sh

      export PATH=$PATH:/home/llvm-project/build/bin
      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/llvm-project/build/lib

6. **Build Triton-distributed**

   Then you can build Triton-distributed.

   .. code-block:: sh

      # Not recommend to use g++
      export CC=clang-19
      export CXX=clang++-19
      # Remove triton installed with torch
      pip uninstall triton
      rm -rf /usr/local/lib/python3.12/dist-packages/triton
      # Install Triton-distributed
      cd /workspace/Triton-distributed
      export USE_TRITON_DISTRIBUTED_AOT=0
      pip3 install -e python --verbose --no-build-isolation

   We also provide AOT version of Triton-distributed. If you want to use AOT (**Not Recommended**), then

   .. code-block:: sh

      cd /workspace/Triton-distributed/
      source scripts/setenv.sh
      bash scripts/gen_aot_code.sh
      export USE_TRITON_DISTRIBUTED_AOT=1
      pip3 install -e python --verbose --no-build-isolation

   .. note:: You have to first build non-AOT version before building AOT version, once you build AOT version, you will always build for AOT in future. To unset this, you have to remove your build directory: `python/build`

7. **Setup environment variables (Optional)**

   .. code-block:: sh

      cd /home/Triton-distributed
      source scripts/setenv.sh

+++++++++++++++++++++++
Test your installation:
+++++++++++++++++++++++

**AllGather GEMM example on single node**

This example runs on a single node with 8 H800 GPUs.

.. code-block:: sh

   bash ./launch.sh ./python/triton_dist/test/nvidia/test_distributed_wait.py --case correctness_tma

**GEMM ReduceScatter example on single node**

This example runs on a single node with 8 H800 GPUs.

.. code-block:: sh

   bash ./launch.sh ./python/triton_dist/test/nvidia/test_gemm_rs.py 8192 8192 29568

**NVSHMEM example in Triton-distributed**

.. code-block:: sh

   bash ./launch.sh ./python/triton_dist/test/nvidia/test_nvshmem_api.py

-------------------------------
Best practice for AMD backend:
-------------------------------


+++++++++++++
Requirements:
+++++++++++++

- ROCM 6.3.0
- Torch 2.4.1 with ROCM support


Starting from the rocm/pytorch:rocm6.1_ubuntu22.04_py3.10_pytorch_2.4 Docker container

+++++++++++++
Steps:
+++++++++++++

1. **Clone the repo**

   .. code-block:: sh

      git clone https://github.com/ByteDance-Seed/Triton-distributed.git

2. **Update submodules**

   .. code-block:: sh

      cd Triton-distributed/
      git submodule update --init --recursive

3. **Install dependencies**

   .. code-block:: sh

      sudo apt-get update -y
      sudo apt install -y libopenmpi-dev
      pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm6.3 --no-deps
      bash ./shmem/rocshmem_bind/build.sh
      python3 -m pip install -i https://test.pypi.org/simple hip-python>=6.3.0 # (or whatever Rocm version you have)
      pip3 install pybind11

4. **Build Triton-distributed**

   .. code-block:: sh

      pip3 install -e python --verbose --no-build-isolation

+++++++++++++++++++++++
Test your installation:
+++++++++++++++++++++++

**GEMM ReduceScatter example on single node**

.. code-block:: sh

   bash ./launch_amd.sh ./python/triton_dist/test/amd/test_ag_gemm_intra_node.py 8192 8192 29568
