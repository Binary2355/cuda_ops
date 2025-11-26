import os
import torch
import torch.distributed as dist
from rdma_ext import rdma_init, rdma_get_unique_id, rdma_put, rdma_get, rdma_get_my_pe, rdma_free, rdma_finalize

def setup():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(local_rank)
    print(f"Rank {rank} (local_rank {local_rank}) initialized on cuda:{local_rank}")

def cleanup():
    dist.destroy_process_group()

def main():
    setup()
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"IMPORTANT:: Rank {rank}, World Size {world_size}")
    
    if rank == 0:
        root_unique_id = rdma_get_unique_id()
        object_list = [root_unique_id]
    else:
        object_list = [None]

    dist.broadcast_object_list(object_list, src=0)
    root_unique_id = object_list[0]

    print(f"Rank {rank}: root_unique_id = {root_unique_id}")
    
    remote_ptr = rdma_init(root_unique_id, rank, world_size)
    
    if remote_ptr is None:
        print(f"Rank {rank}: RDMA initialization failed!")
        return
    
    try:
        current_pe = rdma_get_my_pe()
        print(f"Rank {rank}: Current PE: {current_pe}")
        
        dist.barrier()
        
        # 测试数据交换
        if world_size >= 2:
            if rank == 0:
                my_tensor = torch.ones((100), dtype=torch.float32).cuda() * (rank + 1)
                print(f"Rank 0: Sending tensor to PE 1, tensor mean: {my_tensor.mean().item()}")
                rdma_put(my_tensor, remote_ptr, 1)
                print("Rank 0: Data sent successfully")
                
            elif rank == 1:
                print("Rank 1: Receiving tensor from PE 0")
                tensor_size = 100 * 4  # 100个float32，每个4字节
                received_tensor = rdma_get(remote_ptr, tensor_size, torch.float32, 0)  # 从PE 0接收
                print(f"Rank 1: Data received successfully, tensor shape: {received_tensor.shape}")
                print(f"Rank 1: Tensor mean: {received_tensor.mean().item()}")
                print(f"Rank 1: First few elements: {received_tensor[:5]}")
                
                # 验证数据正确性
                expected_value = 1.0  # 因为rank 0发送的是ones
                if torch.allclose(received_tensor, torch.tensor(expected_value).cuda()):
                    print("Rank 1: Data verification PASSED!")
                else:
                    print("Rank 1: Data verification FAILED!")
            else:
                print(f"Rank {rank}: Not participating in data transfer")
        else:
            print("Need at least 2 ranks for RDMA communication test")
            
    except Exception as e:
        print(f"Rank {rank}: Error during data transfer: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"Rank {rank}: Cleaning up resources")
        rdma_free(remote_ptr)
        rdma_finalize()
        cleanup()

if __name__ == "__main__":
    main()