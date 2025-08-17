"""
最简单的通信kernel示例
基于 tutorials/04-sm-aware-comm-compute-overlap.py 的 communication_kernel
"""

import torch
from triton_dist.utils import nvshmem_free_tensor_sync
from triton_dist.utils import dist_print
import triton
import triton.language as tl
from triton_dist.language.extra import libshmem_device
from triton_dist.utils import initialize_distributed, nvshmem_create_tensors
from triton.language.extra.cuda.language_extra import tid

@triton.jit
def fence_gpu_sc():
    """GPU系统一致性内存栅栏"""
    tl.inline_asm_elementwise(
        asm="fence.gpu.sc;",
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )

@triton.jit
def communication_kernel(
    remote_tensor_ptr,
    local_tensor_ptr,
    communication_signal_ptr,  # 用于通知计算kernel的全局信号缓冲区
    elem_per_rank,
    size_per_elem,
    signal_target,
    local_rank,
    world_size,
    num_comm_sm: tl.constexpr,
):
    """
    通信kernel：执行AllGather并通过global signal通知计算kernel
    
    Args:
        communication_signal_ptr: 全局信号缓冲区，用于通知计算kernel数据准备就绪
    """
    pid = tl.program_id(axis=0)
    thread_idx = tid(0)
    assert pid < num_comm_sm, "程序ID超过通信SM数量"
    
        # 计算需要拉取的ranks数量（除了自己）
    total_pull_ranks = world_size - 1
    
    # 所有通信SM协作，依次从每个源rank拉取数据
    for step in range(total_pull_ranks):
        # 计算当前步骤要拉取的源rank：rank+1, rank+2, ...
        src_rank = (local_rank + step + 1) % world_size
        
        # 计算当前SM在这次拉取中负责的数据块
        # 假设需要128字节对齐
        ALIGNMENT = 128 // size_per_elem
        block_size = tl.cdiv(elem_per_rank, num_comm_sm)
        block_size = tl.cdiv(block_size, ALIGNMENT) * ALIGNMENT  # 确保对齐
        start_elem = pid * block_size
        end_elem = tl.minimum(start_elem + block_size, elem_per_rank)
        
        # 只有当前SM有分配到数据时才执行拉取
        if start_elem < end_elem and end_elem <= elem_per_rank:
            # 计算源和目标地址偏移
            src_rank_offset = src_rank * elem_per_rank
            dst_rank_offset = src_rank * elem_per_rank
            
            # 当前SM负责的具体地址
            src_addr = remote_tensor_ptr + src_rank_offset + start_elem
            dst_addr = local_tensor_ptr + dst_rank_offset + start_elem
            transfer_size = (end_elem - start_elem) * size_per_elem
            
            # 使用getmem_block从远程拉取数据
            libshmem_device.getmem_block(
                dst_addr,      # 本地目标地址
                src_addr,      # 远程源地址  
                transfer_size, # 数据大小
                src_rank,      # 源rank
            )
        
        # 每个通信SM完成自己的部分后，原子递增完成计数器
        completion_counter_ptr = communication_signal_ptr + world_size + src_rank  # 额外空间作为计数器
        
        # 原子递增完成计数
        current_count = tl.atomic_add(completion_counter_ptr, 1) + 1
        if thread_idx == 0:
            # 当所有通信SM都完成时，设置数据就绪信号
            if current_count == num_comm_sm:
                # 使用inline PTX添加内存fence确保数据传输完成后再设置信号
                fence_gpu_sc()
                
                # 设置数据就绪信号
                tl.store(communication_signal_ptr + src_rank, signal_target)


def run_simple_comm_kernel_example():
    """运行简单的通信kernel示例"""
    
    # 初始化分布式环境
        # 初始化分布式环境
    TP_GROUP = initialize_distributed()
    rank = TP_GROUP.rank()
    world_size = TP_GROUP.size()
    
    # 设置参数
    elem_per_rank = 1024
    num_comm_sm = 4
    signal_target = 1
    
    # 创建测试数据
    local_data = torch.randn(elem_per_rank, dtype=torch.float16, device='cuda')
    
    # 创建nvshmem张量
    remote_buffers = nvshmem_create_tensors(
        (elem_per_rank * world_size,),  # 远程缓冲区大小
        torch.float16,
        rank,
        world_size
    )

    remote_tensor = remote_buffers[rank]
    
    # 初始化本地数据到远程缓冲区
    remote_tensor[rank * elem_per_rank:(rank + 1) * elem_per_rank].copy_(local_data)
    
    # 创建通信信号缓冲区
    communication_signal = torch.zeros(world_size * 2, dtype=torch.int32, device="cuda")
    
    print(f"[Rank {rank}] 开始运行简单通信kernel...")
    
    # 启动通信kernel
    grid = (num_comm_sm,)
    communication_kernel[grid](
        remote_tensor,
        remote_tensor,
        communication_signal,
        elem_per_rank,
        2,  # float16 = 2 bytes
        signal_target,
        rank,
        world_size,
        num_comm_sm,
    )
    
    # 等待所有rank完成
    torch.cuda.synchronize()
    
    dist_print(f"[Rank {rank}] signal_buffer: {communication_signal[:world_size].tolist()}", need_sync=True, allowed_ranks=range(world_size))
    nvshmem_free_tensor_sync(remote_tensor)

if __name__ == "__main__":
    run_simple_comm_kernel_example()