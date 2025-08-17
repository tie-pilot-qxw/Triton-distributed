################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
"""
SM感知的通信计算重叠
===================

在这个教程中，您将学习如何实现通信和计算的重叠，同时利用SM occupancy进行优化。

主要特性：

* 通信kernel和persistent GEMM计算kernel协同工作
* 通过global buffer进行信号传递
* SM数量配置：通信SM + 计算SM = 总SM数量
* 测试不同SM分配对整体性能的影响

.. code-block:: bash

    # 运行此教程
    bash ./scripts/launch.sh ./tutorials/04-sm-aware-comm-compute-overlap.py

"""

import os
import time
from typing import List, Dict, Tuple
import json
from dataclasses import dataclass

import nvshmem.core
import torch
from cuda import cuda

import triton
import triton.language as tl
from triton_dist.language.extra import libshmem_device
from triton_dist.utils import (CUDA_CHECK, dist_print, initialize_distributed, nvshmem_barrier_all_on_stream,
                               NVSHMEM_SIGNAL_DTYPE, nvshmem_create_tensors, nvshmem_free_tensor_sync)
from triton.language.extra.cuda.language_extra import __syncthreads, tid
from triton.tools.tensor_descriptor import TensorDescriptor


@dataclass
class SMAllocationConfig:
    """SM分配配置"""
    total_sm: int
    comm_sm: int
    compute_sm: int
    
    def __post_init__(self):
        assert self.comm_sm + self.compute_sm == self.total_sm, \
            f"通信SM({self.comm_sm}) + 计算SM({self.compute_sm}) 必须等于总SM数({self.total_sm})"

@triton.jit
def fence_gpu_sc():
    tl.inline_asm_elementwise(
        asm="fence.gpu.sc;",
        constraints="=r",
        args=[], # 使用一个虚拟输入
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )

# %%
# 通信Kernel：负责AllGather通信并设置信号
@triton.autotune(
    configs=[
        triton.Config(
            {},
            num_warps=32,
        ),
        triton.Config(
            {},
            num_warps=16,
        ),
        triton.Config(
            {},
            num_warps=8,
        ),
        triton.Config(
            {},
            num_warps=4,
        ),
    ],
    key=['num_comm_sm', 'world_size', 'elem_per_rank', 'size_per_elem'],
)
@triton.jit
def communication_kernel(
    remote_tensor_ptr,
    local_tensor_ptr,
    signal_buffer_ptr,
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
            
@triton.jit
def wait_signal(addr, flat_tid):
    if flat_tid == 0:
        tl.inline_asm_elementwise(
            """
            {
                .reg .pred  %p<1>;

                wait_block:
                    ld.global.relaxed.gpu.u32 $0, [$1];
                    setp.eq.u32 %p0, $0, 1;
                    @!%p0 bra wait_block;
            }
            """,
            "=r, l",
            [addr],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )

    tl.inline_asm_elementwise(
        "bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )

@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n

def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K, WS = args["M"], args["N"], args["K"], args.get("WARP_SPECIALIZE", False)
    ws_str = "_ws" if WS else ""
    ret["name"] = f"{kernel.name}{ws_str} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2. * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret

def matmul_tma_persistent_get_configs(pre_hook=None):
    return [
        triton.Config(
            {
                'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": 8, "EPILOGUE_SUBTILE":
                SUBTILE
            }, num_stages=s, num_warps=w, pre_hook=pre_hook)  #
        for BM in [128]  #
        for BN in [128, 256]  #
        for BK in [64, 128]  #
        for s in ([2, 3, 4])  #
        for w in [4, 8]  #
        for SUBTILE in [True, False]  #
    ]

def matmul_tma_set_block_size_hook(nargs):
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", False)
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]
    nargs["a_desc"].block_shape = [BLOCK_M, BLOCK_K]
    nargs["b_desc"].block_shape = [BLOCK_N, BLOCK_K]
    if EPILOGUE_SUBTILE:
        nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N // 2]
    else:
        nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N]


@triton.autotune(
    configs=matmul_tma_persistent_get_configs(pre_hook=matmul_tma_set_block_size_hook),
    key=["M", "N", "K", "world_size"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_tma_persistent(a_desc, b_desc, c_desc,  #
                                 M, N, K,  #
                                 signal_buffer_ptr,
                                 BLOCK_SIZE_M: tl.constexpr,  #
                                 BLOCK_SIZE_N: tl.constexpr,  #
                                 BLOCK_SIZE_K: tl.constexpr,  #
                                 GROUP_SIZE_M: tl.constexpr,  #
                                 FP8_OUTPUT: tl.constexpr,  #
                                 EPILOGUE_SUBTILE: tl.constexpr,  #
                                 NUM_SMS: tl.constexpr,  #
                                 RANK: tl.constexpr,
                                 WORLD_SIZE: tl.constexpr = 8
                                 ):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    # tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Enable warp specialization to leverage async warp scheduling in the GPU.
    # FIXME: This only works on Blackwell right now. On older GPUs, this will
    # use software pipelining.
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        pid_m = (pid_m + RANK * num_pid_m // WORLD_SIZE) % num_pid_m
        # 计算tile对应的数据来源rank
        # 假设数据按rank分布：rank0负责前M/world_size行，rank1负责接下来的M/world_size行...
        rows_per_rank = M // WORLD_SIZE
        required_rank = pid_m * BLOCK_SIZE_M // rows_per_rank
            
        # 等待对应rank的数据准备就绪
        if required_rank != RANK:  # 如果不是本地数据，需要等待通信
            wait_signal(signal_buffer_ptr + required_rank, tid(0))

        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        # tile_id_c += NUM_SMS
        # pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am_c = pid_m * BLOCK_SIZE_M
        offs_bn_c = pid_n * BLOCK_SIZE_N

        # Epilogue subtiling is a technique to break our computation and stores into multiple pieces
        # By subtiling we can reduce shared memory consumption by the epilogue and instead use that
        # memory to increase our stage count.
        # In this case we partition the accumulator into 2 BLOCK_SIZE_M x BLOCK_SIZE_N // 2 tensors
        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)
        else:
            accumulator = accumulator.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], accumulator)


def matmul_tma_persistent(a, b, c, NUM_SMS, signal_buffer: torch.Tensor, rank, world_size):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(
            NUM_SMS,
            triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        ), )

    matmul_kernel_tma_persistent[grid](
        a_desc, b_desc, c_desc,  #
        M, N, K,  #
        signal_buffer,
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        NUM_SMS=NUM_SMS,  #
        RANK=rank,  #
        WORLD_SIZE=world_size,  #
    )
    return c

# %%
# 计算Kernel：持久化GEMM，等待通信kernel的信号
@triton.jit
def persistent_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    communication_signal_ptr,  # 通信信号缓冲区
    M, N, K,
    world_size,
    local_rank,
    num_compute_sm: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
    ready_value: tl.constexpr,
):
    """
    持久化GEMM kernel：等待通信kernel的信号，逐块进行计算
    
    Args:
        communication_signal_ptr: 通信信号缓冲区，用于等待数据准备就绪
    """
    pid = tl.program_id(axis=0)
    
    if pid >= num_compute_sm:
        return
    
    # 计算总的tile数量
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    
    # 计算每个SM负责的tile数量
    tiles_per_sm = total_tiles // num_compute_sm
    remaining_tiles = total_tiles % num_compute_sm
    
    # 当前SM的tile范围
    start_tile = pid * tiles_per_sm + tl.minimum(pid, remaining_tiles)
    end_tile = start_tile + tiles_per_sm
    if pid < remaining_tiles:
        end_tile += 1
    
    # 为每个tile执行GEMM计算
    for tile_idx in range(start_tile, end_tile):
        if tile_idx < total_tiles:
            # 计算tile坐标
            tile_m = tile_idx // num_pid_n
            tile_n = tile_idx % num_pid_n
            
            # 计算tile对应的数据来源rank
            # 假设数据按rank分布：rank0负责前M/world_size行，rank1负责接下来的M/world_size行...
            rows_per_rank = M // world_size
            required_rank = tile_m * BLOCK_SIZE_M // rows_per_rank
            
            # 等待对应rank的数据准备就绪
            if required_rank != local_rank:  # 如果不是本地数据，需要等待通信
                wait_signal(communication_signal_ptr + required_rank, tid(0))
            
            # 执行GEMM计算
            # 计算当前tile的地址偏移
            offs_am = tl.arange(0, BLOCK_SIZE_M) + tile_m * BLOCK_SIZE_M
            offs_bn = tl.arange(0, BLOCK_SIZE_N) + tile_n * BLOCK_SIZE_N
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            
            # 初始化累加器
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            
            # K维度循环
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                k_offs = k * BLOCK_SIZE_K + offs_k
                
                # 加载A矩阵块
                a_ptrs = a_ptr + (offs_am[:, None] * K + k_offs[None, :])
                a_mask = (offs_am[:, None] < M) & (k_offs[None, :] < K)
                a = tl.load(a_ptrs, mask=a_mask, other=0.0)
                
                # 加载B矩阵块
                b_ptrs = b_ptr + (k_offs[:, None] * N + offs_bn[None, :])
                b_mask = (k_offs[:, None] < K) & (offs_bn[None, :] < N)
                b = tl.load(b_ptrs, mask=b_mask, other=0.0)
                
                # 矩阵乘法累加
                accumulator += tl.dot(a, b)
            
            # 存储结果
            c_ptrs = c_ptr + (offs_am[:, None] * N + offs_bn[None, :])
            c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
            tl.store(c_ptrs, accumulator, mask=c_mask)


def run_comm_compute_overlap_benchmark(
    rank: int,
    num_ranks: int,
    local_tensor: torch.Tensor,
    weight: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    signal_buffers: List[torch.Tensor],
    sm_config: SMAllocationConfig,
    iterations: int = 10,
) -> Dict[str, float]:
    """
    运行通信计算重叠基准测试
    
    Returns:
        性能统计数据
    """
    M_per_rank, K = local_tensor.shape
    M = M_per_rank * num_ranks
    N = weight.shape[1]
    
    # 创建通信信号缓冲区（global buffer）
    # 需要额外空间存储完成计数器：前num_ranks个位置存信号，后num_ranks个位置存计数器
    communication_signal = torch.zeros(num_ranks * 2, dtype=torch.int32, device="cuda")
    
    # 创建输出缓冲区
    output = torch.zeros((M, N), dtype=local_tensor.dtype, device="cuda")
    
    # 创建不同的stream实现真正的overlap
    comm_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()
    
    times = []
    
    for iteration in range(iterations + 5):
        # 重置缓冲区
        remote_tensor_buffers[rank].fill_(-1)
        remote_tensor_buffers[rank][
            rank * M_per_rank:(rank + 1) * M_per_rank,
        ].copy_(local_tensor)
        signal_buffers[rank].fill_(0)
        communication_signal.fill_(0)
        output.fill_(0)
        
        # 同步所有ranks
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        
        # 开始计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record(compute_stream)
        
        # 在不同stream上同时启动通信kernel和计算kernel，实现真正的overlap
        
        # 通信kernel - 在comm_stream上运行
        with torch.cuda.stream(comm_stream):
            comm_grid = lambda META: (sm_config.comm_sm,)
            communication_kernel[comm_grid](
                remote_tensor_buffers[rank],  # 远程缓冲区
                remote_tensor_buffers[rank],  # 本地缓冲区
                signal_buffers[rank],         # NVSHMEM信号
                communication_signal,         # 通信信号缓冲区
                M_per_rank * K,
                local_tensor.element_size(),
                1,
                rank,
                num_ranks,
                sm_config.comm_sm,
            )
        
        # 计算kernel - 在compute_stream上运行
        with torch.cuda.stream(compute_stream):
            matmul_tma_persistent(
                remote_tensor_buffers[rank],  # 本地远程缓冲区
                weight,                       # 权重矩阵
                output,                       # 输出缓冲区
                NUM_SMS=sm_config.compute_sm,
                signal_buffer=communication_signal,
                rank=rank,
                world_size=num_ranks,
            )
        
        end_event.record(compute_stream)
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        # 等待所有stream完成
        comm_stream.synchronize()
        compute_stream.synchronize()
        torch.cuda.synchronize()
        dist_print(f"Rank {rank} iteration {iteration} completed.")
        execution_time = start_event.elapsed_time(end_event)
        if iteration >= 5:  # 跳过预热阶段
            times.append(execution_time)
        
        # 同步所有ranks
    
    dist_print(f"Rank {rank} times: {times}", need_sync=True, allowed_ranks=[0])
    # 计算统计数据
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # 计算FLOPs和吞吐量
    flops = 2 * M * N * K  # GEMM的FLOPs
    data_transfer_bytes = (num_ranks - 1) * M_per_rank * K * local_tensor.element_size()
    
    compute_throughput = flops / (avg_time * 1e-3) / 1e12  # TFLOPs/s
    comm_throughput = data_transfer_bytes / (avg_time * 1e-3) / (1024**3)  # GB/s
    
    return {
        'avg_time_ms': avg_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'compute_throughput_tflops': compute_throughput,
        'comm_throughput_gbps': comm_throughput,
    }


def benchmark_standalone_allgather(
    rank: int,
    num_ranks: int,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    signal_buffers: List[torch.Tensor],
    golden_output: torch.Tensor,
    num_comm_sm: int,
    iterations: int = 10,
) -> Dict[str, float]:
    """
    单独测试AllGather性能
    """
    M_per_rank, K = local_tensor.shape
    
    # 创建通信信号缓冲区
    communication_signal = torch.zeros(num_ranks * 2, dtype=torch.int32, device="cuda")
    
    times = []

    for iteration in range(iterations + 5):  # 前5次预热
        # 重置缓冲区
        remote_tensor_buffers[rank].fill_(-1)
        remote_tensor_buffers[rank][
            rank * M_per_rank:(rank + 1) * M_per_rank,
        ].copy_(local_tensor)
        signal_buffers[rank].fill_(0)
        communication_signal.fill_(0)
        
        # 同步所有ranks
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        
        # 开始计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record(torch.cuda.current_stream())
        
        # 只运行通信kernel
        comm_grid = lambda META: (num_comm_sm,)
        communication_kernel[comm_grid](
            remote_tensor_buffers[rank],  # 远程缓冲区
            remote_tensor_buffers[rank],  # 本地缓冲区
            signal_buffers[rank],         # NVSHMEM信号
            communication_signal,         # 通信信号缓冲区
            M_per_rank * K,
            local_tensor.element_size(),
            1,
            rank,
            num_ranks,
            num_comm_sm,
        )
        
        end_event.record(torch.cuda.current_stream())
        
        # 等待完成
        torch.cuda.synchronize()
        execution_time = start_event.elapsed_time(end_event)
        if iteration >= 5:  # 跳过预热阶段
            times.append(execution_time)
        
        # 同步所有ranks
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

        torch.allclose(remote_tensor_buffers[rank], golden_output, atol=1e-3, rtol=1e-3)
    
    dist_print(f"times: {times}", need_sync=True, allowed_ranks=[0])
    # 计算统计数据
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # 计算通信吞吐量
    data_transfer_bytes = (num_ranks - 1) * M_per_rank * K * local_tensor.element_size()
    comm_throughput = data_transfer_bytes / (avg_time * 1e-3) / (1024**3)  # GB/s
    
    return {
        'avg_time_ms': avg_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'comm_throughput_gbps': comm_throughput,
    }


def benchmark_standalone_gemm(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    num_compute_sm: int,
    iterations: int = 10,
) -> Dict[str, float]:
    """
    单独测试GEMM性能（手动设置所有信号为就绪状态）
    """
    M, K = input_tensor.shape
    N = weight.shape[1]
    
    golden = torch.matmul(input_tensor, weight.T)
    # 创建输出缓冲区
    output = torch.zeros((M, N), dtype=input_tensor.dtype, device="cuda")
    
    # 创建通信信号缓冲区，并手动设置所有信号为就绪
    num_ranks = 8  # 假设8个ranks
    communication_signal = torch.ones(num_ranks * 2, dtype=torch.int32, device="cuda")  # 全部设为就绪
    
    times = []
    
    for iteration in range(iterations + 5):  # 前5次预热
        output.fill_(0)
        
        # 开始计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record(torch.cuda.current_stream())
        
        # 只运行计算kernel
        matmul_tma_persistent(input_tensor, weight, output,
                              NUM_SMS=num_compute_sm,
                              signal_buffer=communication_signal,
                              rank=0,  # 本地rank
                              world_size=num_ranks)
        
        end_event.record(torch.cuda.current_stream())
        
        # dist_print(f"Rank {0}: 计算结果验证 - "
                #    f"{'通过' if torch.allclose(golden, output, atol=1e-3, rtol=1e-3) else '失败'}",
                #    need_sync=True, allowed_ranks=[0])
        # 等待完成
        torch.cuda.synchronize()
        execution_time = start_event.elapsed_time(end_event)
        if iteration >= 5:  # 跳过预热阶段
            times.append(execution_time)
    dist_print(f"times: {times}", need_sync=True, allowed_ranks=[0])
    # 计算统计数据
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # 计算计算吞吐量
    flops = 2 * M * N * K  # GEMM的FLOPs
    compute_throughput = flops / (avg_time * 1e-3) / 1e12  # TFLOPs/s
    
    return {
        'avg_time_ms': avg_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'compute_throughput_tflops': compute_throughput,
    }


def test_different_sm_allocations(
    rank: int,
    num_ranks: int,
    local_tensor: torch.Tensor,
    weight: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    signal_buffers: List[torch.Tensor],
    golden_output: torch.Tensor,
    golden_all_gather: torch.Tensor,
    total_sm: int,
    iterations: int = 5,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    测试不同的SM分配策略
    
    Returns:
        不同SM配置的性能结果
    """
    results = {}
    comm_sm_options = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
    comm_sm_options = [sm for sm in comm_sm_options if sm < total_sm]
    
    # 首先运行基准测试
    dist_print(f"Rank {rank}: 运行基准测试...", need_sync=True, allowed_ranks=[0])
    
    # 单独测试AllGather性能
    for comm_sm in comm_sm_options:
        allgather_perf = benchmark_standalone_allgather(
            rank, num_ranks, local_tensor, remote_tensor_buffers, signal_buffers,
            num_comm_sm=comm_sm, iterations=iterations, golden_output=golden_all_gather
        )
        dist_print(f"Rank {rank}: 单独AllGather性能 - 时间={allgather_perf['avg_time_ms']:.3f}ms, "
                  f"吞吐量={allgather_perf['comm_throughput_gbps']:.2f} GB/s",
              need_sync=True, allowed_ranks=[0])
    
    # 单独测试GEMM性能
    M_per_rank, K = local_tensor.shape
    M = M_per_rank * num_ranks
    full_input = torch.empty([M, K], dtype=local_tensor.dtype, device="cuda")
    torch.distributed.all_gather_into_tensor(full_input, local_tensor, group=torch.distributed.group.WORLD)
    
    for comm_sm in comm_sm_options:
        gemm_perf = benchmark_standalone_gemm(
            full_input, weight, num_compute_sm=comm_sm, iterations=iterations
        )
        dist_print(f"Rank {rank}: 单独GEMM性能 - 时间={gemm_perf['avg_time_ms']:.3f}ms, "
              f"吞吐量={gemm_perf['compute_throughput_tflops']:.2f} TFLOPs/s",
              need_sync=True, allowed_ranks=[0])
    
    # 测试不同的SM分配策略
    
    
    dist_print(f"Rank {rank}: 测试不同SM分配策略...", need_sync=True, allowed_ranks=[0])
    
    for comm_sm in comm_sm_options:
        compute_sm = total_sm - comm_sm
        
        if compute_sm <= 0:
            continue
            
        sm_config = SMAllocationConfig(
            total_sm=total_sm,
            comm_sm=comm_sm,
            compute_sm=compute_sm,
        )
        
        dist_print(f"Rank {rank}: 测试配置 - 通信SM: {comm_sm}, 计算SM: {compute_sm}",
                  need_sync=True, allowed_ranks=[0])
        
        try:
            perf_result = run_comm_compute_overlap_benchmark(
                rank, num_ranks, local_tensor, weight,
                remote_tensor_buffers, signal_buffers,
                sm_config, iterations
            )
            
            # 验证结果正确性
            output = torch.zeros_like(golden_output)
            test_overlap_result(rank, num_ranks, local_tensor, weight,
                              remote_tensor_buffers, signal_buffers,
                              sm_config, output)
            
            if torch.allclose(golden_output, output, atol=1e-3, rtol=1e-3):
                msg = f"Rank {rank}: 结果验证通过 ✅"
            else:
                msg = f"Rank {rank}: 结果验证失败 ❌"

            dist_print(msg, need_sync=True, allowed_ranks=range(num_ranks))
            
            results[(comm_sm, compute_sm)] = perf_result
            
            dist_print(f"Rank {rank}: 通信SM={comm_sm}, 计算SM={compute_sm}, "
                      f"时间={perf_result['avg_time_ms']:.3f}ms, "
                      f"计算吞吐量={perf_result['compute_throughput_tflops']:.2f} TFLOPs/s, "
                      f"通信吞吐量={perf_result['comm_throughput_gbps']:.2f} GB/s",
                      need_sync=True, allowed_ranks=[0])
                      
        except Exception as e:
            dist_print(f"Rank {rank}: 配置 ({comm_sm}, {compute_sm}) 测试失败: {e}",
                      need_sync=True, allowed_ranks=[0])
    
    # 添加基准测试结果
    # results[('allgather_only', 0)] = allgather_perf
    # results[(0, 'gemm_only')] = gemm_perf
    
    return results


def test_overlap_result(
    rank: int,
    num_ranks: int,
    local_tensor: torch.Tensor,
    weight: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    signal_buffers: List[torch.Tensor],
    sm_config: SMAllocationConfig,
    output: torch.Tensor,
) -> None:
    """
    执行一次重叠测试并获取结果（用于验证正确性）
    """
    M_per_rank, K = local_tensor.shape
    M = M_per_rank * num_ranks
    N = weight.shape[1]
    
    # 创建通信信号缓冲区
    communication_signal = torch.zeros(num_ranks * 2, dtype=torch.int32, device="cuda")
    
    # 重置缓冲区
    remote_tensor_buffers[rank].fill_(-1)
    remote_tensor_buffers[rank][
        rank * M_per_rank:(rank + 1) * M_per_rank,
    ].copy_(local_tensor)
    signal_buffers[rank].fill_(0)
    communication_signal.fill_(0)
    output.fill_(0)
    
    # 同步所有ranks
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    
    # 创建不同的stream
    comm_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()

    with torch.cuda.stream(comm_stream):
        comm_grid = lambda META: (sm_config.comm_sm,)
        communication_kernel[comm_grid](
                remote_tensor_buffers[rank],
                remote_tensor_buffers[rank],
                signal_buffers[rank],
                communication_signal,
                M_per_rank * K,
                local_tensor.element_size(),
                1,
                rank,
                num_ranks,
                sm_config.comm_sm,
            )
    
    with torch.cuda.stream(compute_stream):
        matmul_tma_persistent(remote_tensor_buffers[rank], weight, output,
                NUM_SMS=sm_config.compute_sm,
                signal_buffer=communication_signal,
                rank=rank,
                world_size=num_ranks,
            )
        # 这里使用持久化GEMM kernel，等待通信kernel的信号
        # 注意：这里的remote_tensor_buffers[rank]已经是AllGather后的完整A矩阵
        # compute_grid = lambda META: (sm_config.compute_sm,)
        # persistent_gemm_kernel[compute_grid](
        #         remote_tensor_buffers[rank],
        #         weight,
        #         output,
        #         communication_signal,
        #         M, N, K,
        #         num_ranks,
        #         rank,
        #         sm_config.compute_sm,
        #         BLOCK_SIZE_M=64,
        #         BLOCK_SIZE_N=64,
        #         BLOCK_SIZE_K=32,
        #         ready_value=1,
        #     )
    
    # 等待完成
    comm_stream.synchronize()
    compute_stream.synchronize()
    torch.cuda.synchronize()

    dist_print(f"communication_signal: {communication_signal}", need_sync=True, allowed_ranks = list(range(num_ranks)))

    # 同步所有ranks
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())


def analyze_sm_allocation_results(
    results: Dict[Tuple[int, int], Dict[str, float]],
    rank: int,
    total_sm: int,
) -> None:
    """
    分析SM分配结果
    """
    if rank != 0:
        return
    
    print("\n" + "="*80)
    print("SM分配策略对通信计算重叠性能的影响")
    print("="*80)
    print(f"总SM数量: {total_sm}")
    print("-"*80)
    print(f"{'通信SM':<8} {'计算SM':<8} {'时间(ms)':<10} {'计算(TFLOPs/s)':<15} {'通信(GB/s)':<12}")
    print("-"*80)
    
    best_overall_time = float('inf')
    best_config = None
    
    for (comm_sm, compute_sm), stats in sorted(results.items()):
        print(f"{comm_sm:<8} {compute_sm:<8} {stats['avg_time_ms']:<10.3f} "
              f"{stats['compute_throughput_tflops']:<15.2f} "
              f"{stats['comm_throughput_gbps']:<12.2f}")
        
        if stats['avg_time_ms'] < best_overall_time:
            best_overall_time = stats['avg_time_ms']
            best_config = (comm_sm, compute_sm)
    
    print("-"*80)
    if best_config:
        print(f"最佳配置: 通信SM={best_config[0]}, 计算SM={best_config[1]}, "
              f"时间={best_overall_time:.3f}ms")
    
    print("\n性能分析:")
    print("- 通信SM太少：通信成为瓶颈，计算SM等待时间长")
    print("- 通信SM太多：计算SM不够，计算吞吐量下降")
    print("- 最优配置需要平衡通信和计算的resource需求")
    print("="*80)

def test_optimized_gemm_single_gpu():
    """
    在单GPU上测试优化的GEMM性能
    """
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    M = 4096 * 4
    K = 2048 * 4
    N = 2048 * 4
    dtype = torch.float16

    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(N, K, device="cuda", dtype=dtype)

    c_gold = torch.matmul(a, b)

    # first benchmark the original GEMM
    for _ in range(10):
        c = torch.empty(M, N, device="cuda", dtype=dtype)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream())
        torch.matmul(a, b, out=c)
        end.record(torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
        flops = 2 * M * N * K / (elapsed_time * 1e-3) / 1e12  # TFLOPs
        print(f"GEMM耗时: {elapsed_time:.3f}ms, FLOPs: {flops:.2f} TFLOPs/s")
        
    config = torch.cuda.get_device_properties(torch.cuda.current_device())
    total_sm = config.multi_processor_count
    print(f"GPU设备: {config.name}, 总SM数量: {total_sm}")
    # now benchmark the optimized GEMM
    for _ in range(10):
        c = torch.empty(M, N, device="cuda", dtype=dtype)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        grid = (total_sm,)
        ranks = 8  # 假设8个rank
        start.record(torch.cuda.current_stream())
        persistent_gemm_kernel[grid](
            a, b, c,
            M, N, K,
            # 0,  # rank
            ranks,
            # torch.ones(ranks, dtype=torch.int32, device="cuda"),  # 信号缓冲区（全部就绪）
            total_sm,
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_N=256,
            BLOCK_SIZE_K=64,
            # GROUP_SIZE_M=8,
            # ready_value=1,
        )
        end.record(torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
        flops = 2 * M * N * K / (elapsed_time * 1e-3) / 1e12  # TFLOPs
        print(f"优化GEMM耗时: {elapsed_time:.3f}ms, FLOPs: {flops:.2f} TFLOPs/s")

        # 验证结果正确性
        if not torch.allclose(c, c_gold, atol=1e-3):
            raise ValueError("优化GEMM结果与参考结果不匹配！")

# test_optimized_gemm_single_gpu()


if __name__ == "__main__":
    # 初始化分布式环境
    TP_GROUP = initialize_distributed()
    rank = TP_GROUP.rank()
    num_ranks = TP_GROUP.size()
    LOCAL_WORLD_SIZE = int(os.getenv("LOCAL_WORLD_SIZE", "8"))
    assert num_ranks == LOCAL_WORLD_SIZE, "此教程设计用于节点内通信"
    
    # 配置测试参数
    M = 4096 * 4  # 总行数
    K = 2048 * 4  # 特征维度
    N = 2048 * 4  # 输出维度
    M_per_rank = M // num_ranks
    dtype = torch.float16
    
    # 获取GPU信息
    device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
    total_sm = device_props.multi_processor_count
    
    dist_print(f"GPU设备: {device_props.name}, 总SM数量: {total_sm}", 
              need_sync=True, allowed_ranks=[0])
    
    # 创建测试数据
    local_data = torch.randn([M_per_rank, K], dtype=dtype, device="cuda")
    weight = torch.randn([N, K], dtype=dtype, device="cuda")
    
    # 创建对称内存缓冲区
    symm_ag_buffers = nvshmem_create_tensors((M, K), dtype, rank, LOCAL_WORLD_SIZE)
    symm_signals = nvshmem_create_tensors((num_ranks,), NVSHMEM_SIGNAL_DTYPE, rank, LOCAL_WORLD_SIZE)
    
    # 计算参考结果
    golden_input = torch.empty([M, K], dtype=dtype, device="cuda")
    torch.distributed.all_gather_into_tensor(golden_input, local_data, group=TP_GROUP)
    # 计算参考输出
    golden_output = torch.matmul(golden_input, weight.T)

    
    # 执行SM分配策略测试
    allocation_results = test_different_sm_allocations(
        rank=rank,
        num_ranks=num_ranks,
        local_tensor=local_data,
        weight=weight,
        remote_tensor_buffers=symm_ag_buffers,
        signal_buffers=symm_signals,
        golden_output=golden_output,
        golden_all_gather=golden_input,
        total_sm=total_sm,
        iterations=5,
    )
    
    # 分析结果
    analyze_sm_allocation_results(allocation_results, rank, total_sm)
    
    # 保存结果
    if rank == 0:
        output_file = f"sm_allocation_results_{device_props.name.replace(' ', '_')}.json"
        with open(output_file, 'w') as f:
            # 转换为可序列化的格式
            serializable_results = {}
            for (comm_sm, compute_sm), stats in allocation_results.items():
                key = f"comm{comm_sm}_compute{compute_sm}"
                serializable_results[key] = {
                    k: v for k, v in stats.items() if k != 'times'
                }
            
            json.dump({
                'device_name': device_props.name,
                'total_sm': total_sm,
                'matrix_sizes': {'M': M, 'K': K, 'N': N},
                'dtype': str(dtype),
                'num_ranks': num_ranks,
                'results': serializable_results,
            }, f, indent=2)
        
        print(f"\n结果已保存到: {output_file}")
    
    # 清理资源
    nvshmem_free_tensor_sync(symm_ag_buffers[rank])
    nvshmem_free_tensor_sync(symm_signals[rank])
    nvshmem.core.finalize()
    
    dist_print(f"Rank {rank}: SM分配策略测试完成！", need_sync=True, allowed_ranks="all")
