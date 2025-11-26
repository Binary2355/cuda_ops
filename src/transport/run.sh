#!/bin/bash

clear
set -e

NPROC_PER_NODE=${MLP_WORKER_GPU:-1}
NNODES=${MLP_WORKER_NUM:-1}
NODE_RANK=${MLP_ROLE_INDEX:-0}
MASTER_ADDR=${MLP_WORKER_0_HOST:-"127.0.0.1"}
MASTER_PORT=${MLP_WORKER_0_PORT:-29800}

rm -rf /file_system/zkl/workspace/src/rdma/nvshmem_ready.flag /file_system/zkl/workspace/src/rdma/nvshmem_unique_id.bin

export NVSHMEM_HOME=/root/miniconda3/lib/python3.10/site-packages/nvidia/nvshmem/
export NVSHMEM_LIB=$NVSHMEM_HOME/lib
export NVSHMEM_INCLUDE=$NVSHMEM_HOME/include
export LD_LIBRARY_PATH=$NVSHMEM_LIB:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvshmem/12/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH


# 在两台机器上都设置
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth1
export NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET
export NVSHMEM_IB_GID_INDEX=7
export NVSHMEM_DEBUG=TRACE

torchrun --nproc_per_node ${NPROC_PER_NODE} \
           --nnodes ${NNODES} \
           --node_rank ${NODE_RANK} \
           --master_addr ${MASTER_ADDR} \
           --master_port ${MASTER_PORT} \
           test.py
