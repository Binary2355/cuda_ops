#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <unistd.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void* allocate_symmetric_memory(const size_t bytes) {
    void* ptr = nvshmem_malloc(bytes);
    TORCH_CHECK(ptr != nullptr, "Failed to allocate symmetric memory");
    return ptr;
}

pybind11::bytearray rdma_get_unique_id() {
    nvshmemx_uniqueid_t unique_id;
    nvshmemx_get_uniqueid(&unique_id);
    std::vector<uint8_t> result(sizeof(nvshmemx_uniqueid_t));
    std::memcpy(result.data(), &unique_id, sizeof(nvshmemx_uniqueid_t));
    return {reinterpret_cast<const char*>(result.data()), result.size()};
}

void* rdma_init(const std::optional<pybind11::bytearray>& root_unique_id_opt, int node_id, int num_nodes) {
    std::vector<uint8_t> root_unique_id_val(root_unique_id_opt->size());
    auto root_unique_id_str = root_unique_id_opt->cast<std::string>();
    std::memcpy(root_unique_id_val.data(), root_unique_id_str.c_str(), root_unique_id_opt->size());

    nvshmemx_uniqueid_t root_unique_id;
    nvshmemx_init_attr_t attr;
    std::memcpy(&root_unique_id, root_unique_id_val.data(), sizeof(nvshmemx_uniqueid_t));
    nvshmemx_set_attr_uniqueid_args(node_id, num_nodes, &root_unique_id, &attr);
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
    nvshmem_barrier_all();
    printf("Node %d: NVSHMEM initialized with %d PEs\n", nvshmem_my_pe(), nvshmem_n_pes());
    
    size_t mem_size = 100 * 1024 * 1024; // 100MB
    void* symmetric_mem = allocate_symmetric_memory(mem_size);

    return symmetric_mem;
}

void rdma_put(const torch::Tensor local_tensor, void* remote_ptr, int peer_pe) {
    CHECK_INPUT(local_tensor);
    nvshmem_putmem(remote_ptr,
                   local_tensor.data_ptr(),
                   local_tensor.nbytes(),
                   peer_pe);
    nvshmem_quiet();
}

torch::Tensor rdma_get(void* remote_ptr, size_t size_in_bytes, torch::ScalarType dtype, int peer_pe) {
    auto options = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
    size_t element_size = torch::elementSize(dtype);
    size_t num_elements = size_in_bytes / element_size;
    torch::Tensor output = torch::empty({(long)num_elements}, options);
    
    nvshmem_getmem(output.data_ptr(), 
                   remote_ptr,
                   size_in_bytes,
                   peer_pe);
    nvshmem_quiet();
    return output;
}

int rdma_get_my_pe()
{
    return nvshmem_my_pe();
}

void rdma_free(void* ptr) {
    if (ptr != nullptr) {
        nvshmem_free(ptr);
    }
}

void rdma_finalize() {
    nvshmem_finalize();
}

PYBIND11_MODULE(rdma_ext, m) {
    m.def("rdma_init", &rdma_init);
    m.def("rdma_get_unique_id", &rdma_get_unique_id);
    m.def("rdma_put", &rdma_put);
    m.def("rdma_get", &rdma_get);
    m.def("rdma_get_my_pe", &rdma_get_my_pe);
    m.def("rdma_free", &rdma_free);
    m.def("rdma_finalize", &rdma_finalize);
}