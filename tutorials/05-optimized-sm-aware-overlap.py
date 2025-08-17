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
优化的SM感知通信计算重叠
=======================

基于07教程的高效实现，优化SM分配和通信计算重叠性能。

主要特性：

* 基于TMA的高效GEMM kernel
* 使用putmem_signal_block的高效AllGather
* SM感知的资源分配
* 准确的性能测试和验证

.. code-block:: bash

    # 运行此教程
    bash ./scripts/launch.sh ./tutorials/05-optimized-sm-aware-overlap.py

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
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
from triton_dist.utils import (CUDA_CHECK, dist_print, initialize_distributed, nvshmem_barrier_all_on_stream,
                               NVSHMEM_SIGNAL_DTYPE, nvshmem_create_tensors, nvshmem_free_tensor_sync)


@dataclass
class SMAllocationConfig:
    """SM分配配置"""
    total_sm: int
    comm_sm: int
    compute_sm: int
    
    def __post_init__(self):
        assert self.comm_sm + self.compute_sm == self.total_sm, \
            f"通信SM({self.comm_sm}) + 计算SM({self.compute_sm}) 必须等于总SM数({self.total_sm})"


# %%
# 高效的AllGather通信Kernel (基于07教程，每个SM对应一个peer)
@triton.jit
def optimized_allgather_kernel(
    ag_buffer_ptr,
    signal_buffer_ptr,
    elem_per_rank,
    size_per_elem,
    signal_target,
    rank,
    world_size,
):
    """
    优化的AllGather kernel，基于07教程的实现
    每个SM负责与一个peer通信
    """
    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)
    
    # 计算需要通信的peer数量
    num_peers = world_size - 1
    
    # 每个SM对应一个peer，如果SM数量超过peer数量则循环分配
    for i in range(pid, num_peers, num_pid):
        # 计算目标peer
        peer = (rank + i + 1) % world_size
        
        # 使用putmem_signal_block发送本rank的数据到目标peer
        libshmem_device.putmem_signal_block(
            ag_buffer_ptr + rank * elem_per_rank,  # 本地数据源
            ag_buffer_ptr + rank * elem_per_rank,  # 远程目标位置
            elem_per_rank * size_per_elem,         # 数据大小
            signal_buffer_ptr + rank,              # 信号位置
            signal_target,                         # 信号值
            libshmem_device.NVSHMEM_SIGNAL_SET,    # 信号操作
            peer,                                  # 目标rank
        )


# %%
# 优化的持久化GEMM Kernel (基于07教程)
def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args[3], args[4], args[5]
    bytes_per_elem = 2  # float16
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


@triton.jit(launch_metadata=_matmul_launch_metadata)
def optimized_persistent_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    rank: tl.constexpr, 
    num_ranks: tl.constexpr,
    ready_ptr,
    num_compute_sm: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ready_value: tl.constexpr = 1,
):
    """
    优化的持久化GEMM kernel，基于07教程的TMA实现
    """
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    # 创建tensor descriptors
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    # 计算每个SM的工作负载
    tiles_per_SM = num_tiles // num_compute_sm
    if start_pid < num_tiles % num_compute_sm:
        tiles_per_SM += 1

    tile_id = start_pid - num_compute_sm
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    M_per_rank = M // num_ranks
    pid_ms_per_rank = tl.cdiv(M_per_rank, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 主计算循环
    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        
        if ki == 0:
            tile_id += num_compute_sm
            
            # 计算tile坐标
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            # swizzle行索引以实现更好的数据局部性
            pid_m = (pid_m + ((rank % num_ranks) * pid_ms_per_rank)) % num_pid_m

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

            # 确定需要的数据来自哪个rank
            rank_beg = offs_am // M_per_rank
            rank_end = (min(offs_am + BLOCK_SIZE_M, M) - 1) // M_per_rank
            
            # 等待相应的数据准备就绪
            token = dl.wait(ready_ptr + rank_beg, rank_end - rank_beg + 1, "gpu", "acquire", waitValue=ready_value)
            a_desc = dl.consume_token(a_desc, token)

        offs_k = ki * BLOCK_SIZE_K
        
        # 执行矩阵乘法
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

        # 存储结果
        if ki == k_tiles - 1:
            c = accumulator.to(dtype)
            c_desc.store([offs_am, offs_bn], c)
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def run_optimized_overlap_benchmark(
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
    运行优化的通信计算重叠基准测试
    """
    M_per_rank, K = local_tensor.shape
    M = M_per_rank * num_ranks
    N = weight.shape[1]
    
    # 创建输出缓冲区
    output = torch.zeros((M, N), dtype=local_tensor.dtype, device="cuda")
    
    # 创建不同的stream实现overlap
    comm_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()
    
    times = []
    
    for iteration in range(iterations):
        # 重置缓冲区
        remote_tensor_buffers[rank].fill_(-1)
        remote_tensor_buffers[rank][
            rank * M_per_rank:(rank + 1) * M_per_rank,
        ].copy_(local_tensor)
        signal_buffers[rank].fill_(0)
        output.fill_(0)
        
        # 同步所有ranks
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        
        # 开始计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record(torch.cuda.current_stream())
        
        # 在不同stream上启动kernels
        # AllGather通信
        with torch.cuda.stream(comm_stream):
            comm_grid = lambda META: (sm_config.comm_sm,)
            optimized_allgather_kernel[comm_grid](
                remote_tensor_buffers[rank],
                signal_buffers[rank],
                M_per_rank * K,
                local_tensor.element_size(),
                1,
                rank,
                num_ranks,
            )
        
        # GEMM计算
        with torch.cuda.stream(compute_stream):
            compute_grid = lambda META: (sm_config.compute_sm,)
            optimized_persistent_gemm_kernel[compute_grid](
                remote_tensor_buffers[rank],  # A矩阵
                weight,                       # B矩阵
                output,                       # C矩阵
                M, N, K,
                rank,
                num_ranks,
                signal_buffers[rank],
                sm_config.compute_sm,
                BLOCK_SIZE_M=128,
                BLOCK_SIZE_N=256,
                BLOCK_SIZE_K=64,
                GROUP_SIZE_M=8,
                ready_value=1,
            )
        
        end_event.record(torch.cuda.current_stream())
        
        # 等待完成
        comm_stream.synchronize()
        compute_stream.synchronize()
        torch.cuda.synchronize()
        execution_time = start_event.elapsed_time(end_event)
        times.append(execution_time)
        
        # 同步所有ranks
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    
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


def benchmark_standalone_optimized_allgather(
    rank: int,
    num_ranks: int,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    signal_buffers: List[torch.Tensor],
    num_comm_sm: int,
    iterations: int = 10,
) -> Dict[str, float]:
    """
    单独测试优化的AllGather性能
    """
    M_per_rank, K = local_tensor.shape
    
    times = []
    
    for iteration in range(iterations + 5):  # 前5次为预热
        # 重置缓冲区
        remote_tensor_buffers[rank].fill_(-1)
        remote_tensor_buffers[rank][
            rank * M_per_rank:(rank + 1) * M_per_rank,
        ].copy_(local_tensor)
        signal_buffers[rank].fill_(0)
        
        # 同步所有ranks
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        
        # 开始计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record(torch.cuda.current_stream())
        
        # 只运行AllGather kernel
        comm_grid = lambda META: (num_comm_sm,)
        optimized_allgather_kernel[comm_grid](
            remote_tensor_buffers[rank],
            signal_buffers[rank],
            M_per_rank * K,
            local_tensor.element_size(),
            1,
            rank,
            num_ranks,
        )
        
        end_event.record(torch.cuda.current_stream())
        
        # 等待完成
        torch.cuda.synchronize()
        execution_time = start_event.elapsed_time(end_event)
        if iteration >= 5:  # 跳过预热阶段
            times.append(execution_time)
        
        # 同步所有ranks
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    
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


def benchmark_standalone_optimized_gemm(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    num_compute_sm: int,
    iterations: int = 10,
) -> Dict[str, float]:
    """
    单独测试优化的GEMM性能（预设所有信号为就绪）
    """
    M, K = input_tensor.shape
    N = weight.shape[1]
    
    # 创建输出缓冲区
    output = torch.zeros((M, N), dtype=input_tensor.dtype, device="cuda")
    
    # 创建信号缓冲区，并设置所有信号为就绪
    num_ranks = 8  # 假设8个ranks
    signal_buffer = torch.ones(num_ranks, dtype=torch.int32, device="cuda")  # 全部设为就绪
    
    times = []
    
    for iteration in range(iterations + 5):  # 前5次为预热
        output.fill_(0)
        
        # 开始计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record(torch.cuda.current_stream())
        
        # 只运行GEMM kernel
        compute_grid = lambda META: (num_compute_sm,)
        optimized_persistent_gemm_kernel[compute_grid](
            input_tensor,         # A矩阵
            weight,               # B矩阵
            output,               # C矩阵
            M, N, K,
            0,                    # rank
            num_ranks,
            signal_buffer,        # 信号缓冲区（全部就绪）
            num_compute_sm,
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_N=256,
            BLOCK_SIZE_K=64,
            GROUP_SIZE_M=8,
            ready_value=1,
        )
        
        end_event.record(torch.cuda.current_stream())
        
        # 等待完成
        torch.cuda.synchronize()
        execution_time = start_event.elapsed_time(end_event)
        if iteration >= 5:  # 跳过预热阶段
            times.append(execution_time)

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


def test_optimized_sm_allocations(
    rank: int,
    num_ranks: int,
    local_tensor: torch.Tensor,
    weight: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    signal_buffers: List[torch.Tensor],
    golden_output: torch.Tensor,
    total_sm: int,
    iterations: int = 5,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    测试优化的SM分配策略
    """
    results = {}

    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    
    # 首先运行基准测试
    dist_print(f"Rank {rank}: 运行优化基准测试...", need_sync=True, allowed_ranks=[0])
    
    # 单独测试AllGather性能
    allgather_perf = benchmark_standalone_optimized_allgather(
        rank, num_ranks, local_tensor, remote_tensor_buffers, signal_buffers,
        num_comm_sm=8, iterations=iterations
    )
    dist_print(f"Rank {rank}: 优化AllGather性能 - 时间={allgather_perf['avg_time_ms']:.3f}ms, "
              f"吞吐量={allgather_perf['comm_throughput_gbps']:.2f} GB/s",
              need_sync=True, allowed_ranks=[0])
    
    # 单独测试GEMM性能
    M_per_rank, K = local_tensor.shape
    M = M_per_rank * num_ranks
    full_input = torch.empty([M, K], dtype=local_tensor.dtype, device="cuda")
    torch.distributed.all_gather_into_tensor(full_input, local_tensor, group=torch.distributed.group.WORLD)
    
    gemm_perf = benchmark_standalone_optimized_gemm(
        full_input, weight, num_compute_sm=total_sm-8, iterations=iterations
    )
    dist_print(f"Rank {rank}: 优化GEMM性能 - 时间={gemm_perf['avg_time_ms']:.3f}ms, "
              f"吞吐量={gemm_perf['compute_throughput_tflops']:.2f} TFLOPs/s",
              need_sync=True, allowed_ranks=[0])
    
    # 测试不同的SM分配策略
    comm_sm_options = [1, 2, 4, 8, 16, 32]
    comm_sm_options = [sm for sm in comm_sm_options if sm < total_sm]
    
    dist_print(f"Rank {rank}: 测试优化SM分配策略...", need_sync=True, allowed_ranks=[0])
    
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
            perf_result = run_optimized_overlap_benchmark(
                rank, num_ranks, local_tensor, weight,
                remote_tensor_buffers, signal_buffers,
                sm_config, iterations
            )
            
            # 验证结果正确性（简单测试）
            test_output = torch.zeros_like(golden_output)
            # TODO: 添加结果验证逻辑
            
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
    results[('allgather_only', 0)] = allgather_perf
    results[(0, 'gemm_only')] = gemm_perf
    
    return results


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
        optimized_persistent_gemm_kernel[grid](
            a, b, c,
            M, N, K,
            0,  # rank
            ranks,
            torch.ones(ranks, dtype=torch.int32, device="cuda"),  # 信号缓冲区（全部就绪）
            total_sm,
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_N=256,
            BLOCK_SIZE_K=64,
            GROUP_SIZE_M=8,
            ready_value=1,
        )
        end.record(torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
        flops = 2 * M * N * K / (elapsed_time * 1e-3) / 1e12  # TFLOPs
        print(f"优化GEMM耗时: {elapsed_time:.3f}ms, FLOPs: {flops:.2f} TFLOPs/s")

        # 验证结果正确性
        if not torch.allclose(c, c_gold, atol=1e-3):
            raise ValueError("优化GEMM结果与参考结果不匹配！")

test_optimized_gemm_single_gpu()

# if __name__ == "__main__":
    # 初始化分布式环境
    # TP_GROUP = initialize_distributed()
    # rank = TP_GROUP.rank()
    # num_ranks = TP_GROUP.size()
    # LOCAL_WORLD_SIZE = int(os.getenv("LOCAL_WORLD_SIZE", "8"))
    # assert num_ranks == LOCAL_WORLD_SIZE, "此教程设计用于节点内通信"
    
    # # 配置测试参数
    # M = 4096 * 4  # 总行数
    # K = 2048 * 4  # 特征维度
    # N = 2048 * 4  # 输出维度
    # M_per_rank = M // num_ranks
    # dtype = torch.float16
    
    # # 获取GPU信息
    # device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
    # total_sm = device_props.multi_processor_count
    
    # dist_print(f"GPU设备: {device_props.name}, 总SM数量: {total_sm}", 
    #           need_sync=True, allowed_ranks=[0])
    
    # # 创建测试数据
    # local_data = torch.randn([M_per_rank, K], dtype=dtype, device="cuda")
    # weight = torch.randn([K, N], dtype=dtype, device="cuda")
    
    # # 创建对称内存缓冲区
    # symm_ag_buffers = nvshmem_create_tensors((M, K), dtype, rank, LOCAL_WORLD_SIZE)
    # symm_signals = nvshmem_create_tensors((num_ranks,), NVSHMEM_SIGNAL_DTYPE, rank, LOCAL_WORLD_SIZE)
    
    # # 计算参考结果
    # golden_input = torch.empty([M, K], dtype=dtype, device="cuda")
    # torch.distributed.all_gather_into_tensor(golden_input, local_data, group=TP_GROUP)
    # golden_output = torch.matmul(golden_input, weight)
    
    # # 执行优化的SM分配策略测试
    # allocation_results = test_optimized_sm_allocations(
    #     rank=rank,
    #     num_ranks=num_ranks,
    #     local_tensor=local_data,
    #     weight=weight,
    #     remote_tensor_buffers=symm_ag_buffers,
    #     signal_buffers=symm_signals,
    #     golden_output=golden_output,
    #     total_sm=total_sm,
    #     iterations=5,
    # )
    
    # # 分析结果
    # if rank == 0:
    #     print("\n" + "="*80)
    #     print("优化SM分配策略性能测试结果")
    #     print("="*80)
    #     print(f"总SM数量: {total_sm}")
    #     print("-"*80)
    #     print(f"{'通信SM':<8} {'计算SM':<8} {'时间(ms)':<10} {'计算(TFLOPs/s)':<15} {'通信(GB/s)':<12}")
    #     print("-"*80)
        
    #     for (comm_sm, compute_sm), stats in sorted(allocation_results.items()):
    #         if isinstance(comm_sm, str) or isinstance(compute_sm, str):
    #             continue
    #         print(f"{comm_sm:<8} {compute_sm:<8} {stats['avg_time_ms']:<10.3f} "
    #               f"{stats['compute_throughput_tflops']:<15.2f} "
    #               f"{stats['comm_throughput_gbps']:<12.2f}")
        
    #     print("="*80)
    
    # # 清理资源
    # nvshmem_free_tensor_sync(symm_ag_buffers[rank])
    # nvshmem_free_tensor_sync(symm_signals[rank])
    # nvshmem.core.finalize()
    # torch.distributed.destroy_process_group()
    
    # dist_print(f"Rank {rank}: 优化测试完成！", need_sync=True, allowed_ranks="all")