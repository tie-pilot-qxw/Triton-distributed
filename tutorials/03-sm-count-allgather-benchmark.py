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
SM数量对AllGather性能影响测试
=============================

在这个教程中，您将测试不同数量的SM对AllGather kernel性能的影响。

主要功能包括：

* 测试不同SM数量配置下的AllGather性能
* 实现pull模式通信：所有SM先从rank+1拉取，然后从rank+2拉取，依次类推
* 收集和分析性能数据
* 对比不同SM配置的吞吐量和延迟

.. code-block:: bash

    # 运行此教程
    bash ./scripts/launch.sh ./tutorials/03-sm-count-allgather-benchmark.py

"""

import os
import time
from typing import List, Dict, Tuple
import json

import nvshmem.core
import torch
from cuda import cuda

import triton
import triton.language as tl
from triton_dist.language.extra import libshmem_device
from triton_dist.utils import (CUDA_CHECK, dist_print, initialize_distributed, nvshmem_barrier_all_on_stream,
                               NVSHMEM_SIGNAL_DTYPE, nvshmem_create_tensors, nvshmem_free_tensor_sync)


# %%
# Pull模式AllGather实现
# 所有SM协作，按照 rank+1, rank+2, ... 的顺序从其他rank拉取数据

@triton.jit
def pull_allgather_kernel_variable_sm(
    remote_tensor_ptr,
    local_tensor_ptr,
    signal_buffer_ptr,
    elem_per_rank,
    size_per_elem,
    signal_target,
    local_rank,
    world_size,
    num_sm: tl.constexpr,
):
    """
    使用指定数量SM的pull模式AllGather kernel
    
    所有SM协作从其他ranks拉取数据：
    - 先从rank+1拉取
    - 然后从rank+2拉取
    - 依次类推直到所有ranks
    
    参数:
        num_sm: 使用的SM数量
    """
    pid = tl.program_id(axis=0)
    
    if pid < num_sm:  # 只使用指定数量的SM
        # 计算需要拉取的ranks数量（除了自己）
        total_pull_ranks = world_size - 1
        
        # 所有SM协作，依次从每个源rank拉取数据
        for step in range(total_pull_ranks):
            # 计算当前步骤要拉取的源rank：rank+1, rank+2, ...
            src_rank = (local_rank + step + 1) % world_size
            
            # 计算当前SM在这次拉取中负责的数据块
            # 将每个rank的数据分成num_sm个块，每个SM负责一块
            block_size = elem_per_rank // num_sm
            remaining_elems = elem_per_rank % num_sm
            
            # 当前SM负责的数据范围
            start_elem = pid * block_size + tl.minimum(pid, remaining_elems)
            end_elem = start_elem + block_size
            if pid < remaining_elems:
                end_elem += 1
            
            # 只有当前SM有分配到数据时才执行拉取
            if start_elem < end_elem and end_elem <= elem_per_rank:
                # 计算源和目标地址偏移
                src_rank_offset = src_rank * elem_per_rank
                dst_rank_offset = src_rank * elem_per_rank
                
                # 当前SM负责的具体地址
                src_addr = remote_tensor_ptr + src_rank_offset + start_elem
                dst_addr = local_tensor_ptr + dst_rank_offset + start_elem
                transfer_size = (end_elem - start_elem) * size_per_elem
                
                # 使用getmem_block从远程拉取数据（block级别API）
                libshmem_device.getmem_block(
                    dst_addr,      # 本地目标地址
                    src_addr,      # 远程源地址
                    transfer_size, # 数据大小
                    src_rank,      # 源rank
                )


def pull_allgather_with_sm_count(
    rank: int,
    num_ranks: int,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    signal_buffers: List[torch.Tensor],
    stream: torch.cuda.Stream,
    num_sm: int,
) -> float:
    """
    使用指定数量SM执行pull模式AllGather并测量性能
    
    返回:
        执行时间（毫秒）
    """
    M_per_rank, N = local_tensor.shape
    
    # 重置缓冲区
    remote_tensor_buffers[rank].fill_(-1)
    # 将本地数据复制到对称内存的对应位置
    remote_tensor_buffers[rank][
        rank * M_per_rank:(rank + 1) * M_per_rank,
    ].copy_(local_tensor)
    signal_buffers[rank].fill_(0)
    
    # 同步所有ranks，确保数据已准备好
    nvshmem_barrier_all_on_stream(stream)
    
    # 开始计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    with torch.cuda.stream(stream):
        start_event.record(stream)
        
        # 启动kernel，使用指定数量的SM进行pull操作
        grid = lambda META: (num_sm,)
        pull_allgather_kernel_variable_sm[grid](
            remote_tensor_buffers[rank],  # 远程缓冲区（对称内存）
            remote_tensor_buffers[rank],  # 本地缓冲区（同一个，用于接收数据）
            signal_buffers[rank],
            M_per_rank * N,
            local_tensor.element_size(),
            1,
            rank,
            num_ranks,
            num_sm,
            num_warps=4
        )
        
        end_event.record(stream)
    
    # 等待所有操作完成
    torch.cuda.synchronize()
    execution_time = start_event.elapsed_time(end_event)
    
    # 同步确保所有ranks完成
    nvshmem_barrier_all_on_stream(stream)
    dist_print(f"Rank {rank}: 使用 {num_sm} 个SM完成拉取，耗时 {execution_time:.3f} ms",)
    
    return execution_time


def benchmark_sm_configurations(
    rank: int,
    num_ranks: int,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    signal_buffers: List[torch.Tensor],
    golden: torch.Tensor,
    max_sm: int = 128,
    iterations: int = 10,
) -> Dict[int, Dict[str, float]]:
    """
    测试不同SM配置的性能
    
    参数:
        max_sm: 最大SM数量
        iterations: 每个配置的测试迭代次数
    
    返回:
        性能统计数据字典
    """
    # SM数量配置列表
    sm_configs = [1, 2, 4, 8, 16, 32, 64]
    if max_sm > 64:
        sm_configs.extend([128, 256])
    
    # 过滤超过最大SM数量的配置
    sm_configs = [sm for sm in sm_configs if sm <= max_sm]
    
    results = {}
    stream = torch.cuda.current_stream()
    
    dist_print(f"Rank {rank}: 开始SM数量性能测试 (Pull模式)...", need_sync=True, allowed_ranks=[0])
    
    for num_sm in sm_configs:
        dist_print(f"Rank {rank}: 测试 {num_sm} 个SM配置...", need_sync=True, allowed_ranks=[0])
        
        times = []
        
        # 预热
        for _ in range(2):
            pull_allgather_with_sm_count(
                rank, num_ranks, local_tensor, remote_tensor_buffers,
                signal_buffers, stream, num_sm
            )
        
        # 正式测试
        for i in range(iterations):
            exec_time = pull_allgather_with_sm_count(
                rank, num_ranks, local_tensor, remote_tensor_buffers,
                signal_buffers, stream, num_sm
            )
            times.append(exec_time)
            
            # 验证结果正确性
            if not torch.allclose(golden, remote_tensor_buffers[rank], atol=1e-5, rtol=1e-5):
                dist_print(f"Rank {rank}: 错误！SM={num_sm}, 迭代={i} 结果不正确", 
                          need_sync=True, allowed_ranks="all")
                dist_print(f"期望形状: {golden.shape}, 实际形状: {remote_tensor_buffers[rank].shape}")
                break
        
        # 计算统计数据
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # 计算吞吐量 (GB/s)
        # 总数据量 = 每个rank的数据 * ranks数量
        data_size_gb = (local_tensor.numel() * local_tensor.element_size() * num_ranks) / (1024**3)
        throughput = data_size_gb / (avg_time / 1000)  # 转换为秒
        
        results[num_sm] = {
            'avg_time_ms': avg_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'throughput_gbps': throughput,
            'times': times,
        }
        
        dist_print(f"Rank {rank}: SM={num_sm:3d}, 平均时间={avg_time:.3f}ms, "
                  f"吞吐量={throughput:.2f} GB/s", need_sync=True, allowed_ranks=[0])
    
    return results


def analyze_and_report_results(
    results: Dict[int, Dict[str, float]],
    rank: int,
    tensor_size: Tuple[int, int],
    dtype: torch.dtype,
    num_ranks: int,
) -> None:
    """
    分析并报告性能测试结果
    """
    if rank != 0:
        return
    
    print("\n" + "="*80)
    print("SM数量对Pull模式AllGather性能影响测试结果")
    print("="*80)
    print(f"张量尺寸: {tensor_size}")
    print(f"数据类型: {dtype}")
    print(f"Rank数量: {num_ranks}")
    print(f"通信模式: Pull模式 (所有SM协作从rank+1, rank+2, ... 依次拉取)")
    print("-"*80)
    print(f"{'SM数量':<8} {'平均时间(ms)':<12} {'最小时间(ms)':<12} {'最大时间(ms)':<12} {'吞吐量(GB/s)':<12}")
    print("-"*80)
    
    best_throughput = 0
    best_sm_count = 0
    
    for num_sm in sorted(results.keys()):
        stats = results[num_sm]
        print(f"{num_sm:<8} {stats['avg_time_ms']:<12.3f} {stats['min_time_ms']:<12.3f} "
              f"{stats['max_time_ms']:<12.3f} {stats['throughput_gbps']:<12.2f}")
        
        if stats['throughput_gbps'] > best_throughput:
            best_throughput = stats['throughput_gbps']
            best_sm_count = num_sm
    
    print("-"*80)
    print(f"最佳配置: {best_sm_count} 个SM, 吞吐量: {best_throughput:.2f} GB/s")
    
    # 分析性能趋势
    print("\n性能分析:")
    sm_counts = sorted(results.keys())
    throughputs = [results[sm]['throughput_gbps'] for sm in sm_counts]
    
    # 分析扩展性
    baseline_throughput = throughputs[0]  # 1个SM的性能
    
    print(f"- 基准性能 (1 SM): {baseline_throughput:.2f} GB/s")
    
    for i, sm_count in enumerate(sm_counts[1:], 1):
        actual_speedup = throughputs[i] / baseline_throughput
        ideal_speedup = sm_count / sm_counts[0]
        efficiency = (actual_speedup / ideal_speedup) * 100
        
        print(f"- {sm_count:3d} SM: {throughputs[i]:.2f} GB/s, "
              f"加速比 {actual_speedup:.2f}x (理想 {ideal_speedup:.2f}x), "
              f"效率 {efficiency:.1f}%")
        
        if efficiency < 70:
            print(f"  ⚠️  效率下降，可能存在通信瓶颈或SM利用率问题")
    
    # 找到最佳效率点
    max_efficiency = 0
    best_efficiency_sm = sm_counts[0]
    
    for i, sm_count in enumerate(sm_counts):
        if i == 0:
            efficiency = 100.0
        else:
            actual_speedup = throughputs[i] / baseline_throughput
            ideal_speedup = sm_count / sm_counts[0]
            efficiency = (actual_speedup / ideal_speedup) * 100
        
        if efficiency > max_efficiency:
            max_efficiency = efficiency
            best_efficiency_sm = sm_count
    
    print(f"\n- 最高效率配置: {best_efficiency_sm} SM ({max_efficiency:.1f}% 效率)")
    print("="*80)


if __name__ == "__main__":
    # 初始化分布式环境
    TP_GROUP = initialize_distributed()
    rank = TP_GROUP.rank()
    num_ranks = TP_GROUP.size()
    LOCAL_WORLD_SIZE = int(os.getenv("LOCAL_WORLD_SIZE"))
    assert num_ranks == LOCAL_WORLD_SIZE, "此教程设计用于节点内通信"
    
    # 配置测试参数
    M = 8192
    N = 12288
    M_per_rank = M // num_ranks
    dtype = torch.float16
    
    # 获取GPU信息
    device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
    max_sm = device_props.multi_processor_count
    
    dist_print(f"GPU设备: {device_props.name}, SM数量: {max_sm}", 
              need_sync=True, allowed_ranks=[0])
    
    # 创建测试数据
    local_data = torch.randn([M_per_rank, N], dtype=dtype, device="cuda")
    symm_ag_buffers = nvshmem_create_tensors((M, N), dtype, rank, LOCAL_WORLD_SIZE)
    symm_signals = nvshmem_create_tensors((num_ranks,), NVSHMEM_SIGNAL_DTYPE, rank, LOCAL_WORLD_SIZE)
    
    # 计算参考结果
    golden = torch.empty([M, N], dtype=dtype, device="cuda")
    torch.distributed.all_gather_into_tensor(golden, local_data, group=TP_GROUP)
    
    # 执行性能测试
    benchmark_results = benchmark_sm_configurations(
        rank=rank,
        num_ranks=num_ranks,
        local_tensor=local_data,
        remote_tensor_buffers=symm_ag_buffers,
        signal_buffers=symm_signals,
        golden=golden,
        max_sm=max_sm,
        iterations=10,
    )
    
    # 分析并报告结果
    analyze_and_report_results(
        results=benchmark_results,
        rank=rank,
        tensor_size=(M, N),
        dtype=dtype,
        num_ranks=num_ranks,
    )
    
    # 保存结果到文件
    if rank == 0:
        output_file = f"pull_allgather_sm_benchmark_{device_props.name.replace(' ', '_')}.json"
        with open(output_file, 'w') as f:
            # 转换为可序列化的格式
            serializable_results = {}
            for sm_count, stats in benchmark_results.items():
                serializable_results[str(sm_count)] = {
                    k: v for k, v in stats.items() if k != 'times'  # 排除详细时间数据
                }
            
            json.dump({
                'device_name': device_props.name,
                'max_sm': max_sm,
                'tensor_size': [M, N],
                'dtype': str(dtype),
                'num_ranks': num_ranks,
                'communication_mode': 'pull',
                'results': serializable_results,
            }, f, indent=2)
        
        print(f"\n结果已保存到: {output_file}")
    
    # 清理资源
    nvshmem_free_tensor_sync(symm_ag_buffers[rank])
    nvshmem_free_tensor_sync(symm_signals[rank])
    nvshmem.core.finalize()
    
    dist_print(f"Rank {rank}: 测试完成！", need_sync=True, allowed_ranks="all")
    torch.distributed.destroy_process_group()