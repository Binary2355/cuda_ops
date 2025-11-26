#/bin/bash

clear
set -e

rm -rf /file_system/zkl/workspace/src/rdma/nvshmem_ready.flag /file_system/zkl/workspace/src/rdma/nvshmem_unique_id.bin

export NVSHMEM_HOME=/root/miniconda3/lib/python3.10/site-packages/nvidia/nvshmem/
export NVSHMEM_LIB=$NVSHMEM_HOME/lib
export NVSHMEM_INCLUDE=$NVSHMEM_HOME/include

pip install -v .
