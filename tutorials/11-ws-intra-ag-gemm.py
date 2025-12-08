import pytest
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
import triton_dist.language as dl
from typing import Optional
from triton.tools.tensor_descriptor import TensorDescriptor
import os
from typing import List

import nvshmem.core
import torch
from cuda import cuda

import triton
import triton.language as tl
from triton_dist.language.extra import libshmem_device
from triton_dist.utils import (CUDA_CHECK, dist_print, initialize_distributed,
                               nvshmem_barrier_all_on_stream,
                               NVSHMEM_SIGNAL_DTYPE, nvshmem_create_tensors,
                               nvshmem_free_tensor_sync, perf_func)


DEVICE = "cuda"
M, N, K = (8192, 8192, 8192)


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip_cdna2():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'


def alloc_fn(size: int, align: int, stream: Optional[int]):
    assert align == 128
    assert stream == 0
    return torch.empty(size, dtype=torch.int8, device=DEVICE)

def cp_engine_producer_all_gather_full_mesh_pull(
    rank,
    num_ranks,
    num_splits,
    remote_tensor_buffers: List[torch.Tensor],
    ag_stream: torch.cuda.Stream,
    barrier_buffer: torch.Tensor,
    ready_value=1,
):
    """All gather using copy engine in pull mode."""
    
    M, _ = remote_tensor_buffers[0].shape
    assert M % num_ranks == 0, "M must be divisible by num_ranks"
    M_per_rank = M // num_ranks
    assert M_per_rank % num_splits == 0, "M_per_rank must be divisible by num_splits"
    M_per_split = M_per_rank // num_splits

    rank_orders = [(rank + i) % num_ranks for i in range(num_ranks)]

    with torch.cuda.stream(ag_stream):
        for src_rank in rank_orders:
            if src_rank == rank:
                continue

            for split_id in range(num_splits):
                start_idx = src_rank * M_per_rank + split_id * M_per_split
                end_idx = start_idx + M_per_split
                # peer: src_rank, offset src_rank[src_rank] -> rank[src_rank]
                dst = remote_tensor_buffers[rank][start_idx : end_idx, :]
                src = remote_tensor_buffers[src_rank][start_idx : end_idx, :]
                dst.copy_(src)
                (err, ) = cuda.cuStreamWriteValue32(
                    ag_stream.cuda_stream,
                    barrier_buffer[src_rank * num_splits + split_id].data_ptr(),
                    ready_value,
                    cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
                )
                CUDA_CHECK(err)


def matmul_tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BM"]
    BLOCK_N = nargs["BN"]
    BLOCK_K = nargs["BK"]
    NUM_MMA_GROUPS = nargs["NUM_MMA_GROUPS"]
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    nargs["a_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_K]
    nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N]
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", False)
    if EPILOGUE_SUBTILE:
        nargs["c_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_N // 2]
    else:
        nargs["c_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_N]


def matmul_get_configs():
    return [
        triton.Config({'BM': BM, 'BN': BN, "BK": BK, "GROUP_SIZE_M": 8, "NUM_STAGES": num_stage,
                "NUM_MMA_WARPS": 8,
                "NUM_MMA_GROUPS": 2,
                "EPILOGUE_SUBTILE": True,}, 
                      num_stages=1, num_warps=4, pre_hook=matmul_tma_set_block_size_hook) \
        for BM in [128, 256] \
        for BN in [128, 256] \
        for BK in [64, 128] \
        for num_stage in [2, 3, 4, 5]
    ]

@triton.autotune(
    # Autotune configs can be reused or adapted
    configs=matmul_get_configs(),
    key=["M", "N", "K"],
    use_cuda_graph=True,
)
@triton.jit
def matmul_kernel_tlx_ws_persistent(
    a_desc, b_desc, c_desc, ready_ptr,
    M, N, K, ready_value,
    NUM_SMS: tl.constexpr, # 1. 新增：传入SM数量
    COMM_BLOCK_SIZE_M: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_MMA_WARPS: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
):
    BLOCK_M_SPLIT: tl.constexpr = BM // NUM_MMA_GROUPS
    NUM_COMM_BLOCKS = M // COMM_BLOCK_SIZE_M
    NUM_COMM_BLOCKS_PER_RANK = NUM_COMM_BLOCKS // WORLD_SIZE
    tl.static_assert(COMM_BLOCK_SIZE_M % BM == 0, "COMM_BLOCK_SIZE_M must be multiple of BM")
    NUM_PID_M_PER_COMM_BLOCK = COMM_BLOCK_SIZE_M // BM

    a = tlx.local_alloc((BLOCK_M_SPLIT, BK), tlx.dtype_of(a_desc), NUM_STAGES * NUM_MMA_GROUPS)
    b = tlx.local_alloc((BK, BN), tlx.dtype_of(b_desc), NUM_STAGES)
    bars_empty_a = tlx.alloc_barriers(num_barriers=NUM_STAGES * NUM_MMA_GROUPS, arrive_count=1)
    bars_full_a = tlx.alloc_barriers(num_barriers=NUM_STAGES * NUM_MMA_GROUPS, arrive_count=1)
    bars_empty_b = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=NUM_MMA_GROUPS)
    bars_full_b = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)

    with tlx.async_tasks():
        # Producer (async load)
        with tlx.async_task("default"):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_tiles = num_pid_m * num_pid_n
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            
            p = 1
            buf = 0
            
            for tile_id in range(start_pid, num_tiles, NUM_SMS):
                group_id = tile_id // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
                pid_n = (tile_id % num_pid_in_group) // group_size_m

                pid_m = (pid_m + NUM_PID_M_PER_COMM_BLOCK * RANK * NUM_COMM_BLOCKS_PER_RANK) % num_pid_m
                
                comm_block_id = pid_m // NUM_PID_M_PER_COMM_BLOCK
                if comm_block_id // NUM_COMM_BLOCKS_PER_RANK != RANK:
                    # wait for data to arrive
                    token = dl.wait(ready_ptr + comm_block_id, 1, "gpu", "acquire", waitValue=ready_value)
                    a_desc = dl.consume_token(a_desc, token)

                offset_am = pid_m * BM
                offset_bn = pid_n * BN

                for k in range(0, tl.cdiv(K, BK)):
                    offset_k = k * BK
                    
                    # Async load to a[buf]
                    empty_a_1st = tlx.local_view(bars_empty_a, buf)
                    full_a_1st = tlx.local_view(bars_full_a, buf)
                    tlx.barrier_wait(bar=empty_a_1st, phase=p)
                    tlx.barrier_expect_bytes(full_a_1st, BLOCK_M_SPLIT * BK * 2)
                    data_a_1st = tlx.local_view(a, buf)
                    tlx.async_descriptor_load(a_desc, data_a_1st, [offset_am, offset_k], full_a_1st)

                    # Async load to b[buf]
                    empty_b = tlx.local_view(bars_empty_b, buf)
                    full_b = tlx.local_view(bars_full_b, buf)
                    tlx.barrier_wait(bar=empty_b, phase=p)
                    tlx.barrier_expect_bytes(full_b, BN * BK * 2)
                    data_b = tlx.local_view(b, buf)
                    tlx.async_descriptor_load(b_desc, data_b, [offset_k, offset_bn], full_b)

                    # Async load to a[buf+NUM_STAGES]
                    empty_a_2nd = tlx.local_view(bars_empty_a, buf + NUM_STAGES)
                    full_a_2nd = tlx.local_view(bars_full_a, buf + NUM_STAGES)
                    tlx.barrier_wait(bar=empty_a_2nd, phase=p)
                    tlx.barrier_expect_bytes(bar=full_a_2nd, size=BLOCK_M_SPLIT * BK * 2)
                    data_a_2nd = tlx.local_view(a, buf + NUM_STAGES)
                    tlx.async_descriptor_load(a_desc, data_a_2nd, [offset_am + BLOCK_M_SPLIT, offset_k], full_a_2nd)

                    p = p ^ (buf == (NUM_STAGES - 1))
                    buf = (buf + 1) % NUM_STAGES

        # Consumers (wgmma + async store)
        with tlx.async_task(num_warps=4, replicate=2, registers=232):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_tiles = num_pid_m * num_pid_n
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            cid: tl.constexpr = tlx.async_task_replica_id()

            p = 0
            buf = 0
            
            for tile_id in range(start_pid, num_tiles, NUM_SMS):
                group_id = tile_id // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
                pid_n = (tile_id % num_pid_in_group) // group_size_m
                pid_m = (pid_m + NUM_PID_M_PER_COMM_BLOCK * RANK * NUM_COMM_BLOCKS_PER_RANK) % num_pid_m
                
                offset_am = pid_m * BM
                offset_bn = pid_n * BN

                acc = tl.zeros([BM // 2, BN], dtype=tl.float32)

                last_buf = buf
                full_a = tlx.local_view(bars_full_a, buf + NUM_STAGES * tlx.async_task_replica_id())
                full_b = tlx.local_view(bars_full_b, buf)
                tlx.barrier_wait(bar=full_a, phase=p)
                tlx.barrier_wait(bar=full_b, phase=p)

                data_a = tlx.local_view(a, buf + NUM_STAGES * tlx.async_task_replica_id())
                data_b = tlx.local_view(b, buf)

                acc = tlx.async_dot(data_a, data_b, acc)

                p = p ^ (buf == (NUM_STAGES - 1))
                buf = (buf + 1) % NUM_STAGES

                for k in range(1, tl.cdiv(K, BK)):
                    
                    full_a = tlx.local_view(bars_full_a, buf + NUM_STAGES * tlx.async_task_replica_id())
                    full_b = tlx.local_view(bars_full_b, buf)
                    tlx.barrier_wait(bar=full_a, phase=p)
                    tlx.barrier_wait(bar=full_b, phase=p)

                    data_a = tlx.local_view(a, buf + NUM_STAGES * tlx.async_task_replica_id())
                    data_b = tlx.local_view(b, buf)

                    acc = tlx.async_dot(data_a, data_b, acc)
                    acc = tlx.async_dot_wait(1, acc)

                    empty_a = tlx.local_view(bars_empty_a, last_buf + NUM_STAGES * tlx.async_task_replica_id())
                    empty_b = tlx.local_view(bars_empty_b, last_buf)
                    tlx.barrier_arrive(empty_a)
                    tlx.barrier_arrive(empty_b)

                    last_buf = buf
                    p = p ^ (buf == (NUM_STAGES - 1))
                    buf = (buf + 1) % NUM_STAGES

                offset_cm = offset_am + BLOCK_M_SPLIT * tlx.async_task_replica_id()

                acc = tlx.async_dot_wait(0, acc)
                empty_a = tlx.local_view(bars_empty_a, last_buf + NUM_STAGES * tlx.async_task_replica_id())
                empty_b = tlx.local_view(bars_empty_b, last_buf)
                tlx.barrier_arrive(empty_a)
                tlx.barrier_arrive(empty_b)

                if EPILOGUE_SUBTILE:
                    acc = tl.reshape(acc, (BLOCK_M_SPLIT, 2, BN // 2))
                    acc = tl.permute(acc, (0, 2, 1))
                    acc0, acc1 = tl.split(acc)
                    c0 = acc0.to(tlx.dtype_of(c_desc))
                    c_desc.store([offset_cm, offset_bn], c0)
                    c1 = acc1.to(tlx.dtype_of(c_desc))
                    c_desc.store([offset_cm, offset_bn + BN // 2], c1)
                else:
                    c_desc.store([offset_cm, offset_bn], acc.to(tlx.dtype_of(c_desc)))

def matmul_tlx_ws_persistent_ag(a_list: List[torch.Tensor], b, ready_ptr, world_size, rank, COMM_SPLIT=1):
    if not hasattr(matmul_tlx_ws_persistent_ag, "ready_value"):
        matmul_tlx_ws_persistent_ag.ready_value = 1  # 仅在第一次调用时初始化
    else:
        matmul_tlx_ws_persistent_ag.ready_value ^= 1  # 每次调用切换ready_value

    a = a_list[rank]
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Illegal dimensions of input operands"
    assert a.is_contiguous(), "Matrix A must be contiguous"

    (M, N, K) = (a.shape[0], b.shape[1], a.shape[1])
    c = torch.zeros((M, N), dtype=torch.float16, device=DEVICE)

    M_per_rank = M // world_size
    assert M % world_size == 0, "M must be divisible by world_size"
    M_per_comm = M_per_rank // COMM_SPLIT
    assert M_per_rank % COMM_SPLIT == 0, "M_per_rank must be divisible by COMM_SPLIT"

    # 获取设备上的SM数量
    NUM_SMS = torch.cuda.get_device_properties(DEVICE).multi_processor_count

    dummy_block = [1, 1]
    desc_in_1 = TensorDescriptor(a, shape=[M, K], strides=[K, 1], block_shape=dummy_block)
    desc_in_2 = TensorDescriptor(b, shape=[K, N], strides=[N, 1], block_shape=dummy_block)
    desc_out = TensorDescriptor(c, shape=[M, N], strides=[N, 1], block_shape=dummy_block)

    # 5. 修改Grid计算方式
    def grid(META):
        # 启动的线程块数是 SM数量 和 总块数 中的较小值
        num_m_blocks = triton.cdiv(M, META['BM'])
        num_n_blocks = triton.cdiv(N, META['BN'])
        total_blocks = num_m_blocks * num_n_blocks
        return (min(NUM_SMS, total_blocks),)
    
    gemm_stream = torch.cuda.current_stream()
    ag_stream = torch.cuda.Stream()

    nvshmem_barrier_all_on_stream(gemm_stream)
    ag_stream.wait_stream(gemm_stream)

    cp_engine_producer_all_gather_full_mesh_pull(rank, world_size, COMM_SPLIT, a_list, ag_stream, ready_ptr, matmul_tlx_ws_persistent_ag.ready_value)

    with torch.cuda.stream(gemm_stream):
        matmul_kernel_tlx_ws_persistent[grid](
            desc_in_1, desc_in_2, desc_out, ready_ptr,
            M, N, K, matmul_tlx_ws_persistent_ag.ready_value, 
            NUM_SMS=NUM_SMS, # 传入SM数量 
            COMM_BLOCK_SIZE_M=M_per_comm,
            WORLD_SIZE=world_size,
            RANK=rank,
        )

    gemm_stream.wait_stream(ag_stream)
    return c

TORCH_HAS_FP8 = False

ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'

# Benchmarking
configs = []
for fp8_inputs in [False, True]:
    if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
        continue
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
            x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=["triton"] if fp8_inputs else [ref_lib.lower(), "triton"],  # Label name for the lines
            line_names=["Triton"] if fp8_inputs else [ref_lib, "Triton"],  # Line styles
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="matmul-performance-" +
            ("fp16" if not fp8_inputs else "fp8"),  # Name for the plot, used also as a file name for saving the plot.
            args={"fp8_inputs": fp8_inputs},
        ))

@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider, fp8_inputs):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    if TORCH_HAS_FP8 and fp8_inputs:
        a = a.to(torch.float8_e5m2)
        b = b.T
        b = b.to(torch.float8_e5m2)
    quantiles = [0.5, 0.2, 0.8]
    if provider == ref_lib.lower():
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

def torch_ag_gemm(a_full, a_local, b, group):
    torch.distributed.all_gather_into_tensor(a_full, a_local, group=group)
    return torch.matmul(a_full, b)

if __name__ == "__main__":
    TP_GROUP = initialize_distributed()
    rank = TP_GROUP.rank()
    num_ranks = TP_GROUP.size()
    LOCAL_WORLD_SIZE = int(os.getenv("LOCAL_WORLD_SIZE"))
    assert num_ranks == LOCAL_WORLD_SIZE, "This tutorial is designed for intra-node"

    M = 4096 * 8
    N = 6656
    K = 16384
    M_per_rank = M // num_ranks
    COMM_SPLIT = 1
    dtype = torch.float16

    local_a = torch.randn([M_per_rank, K], dtype=dtype, device="cuda")
    b = torch.randn([K, N], dtype=dtype, device="cuda")
    a_symm_buffers = nvshmem_create_tensors((M, K), dtype, rank,
                                             LOCAL_WORLD_SIZE)
    a_symm_buffer = a_symm_buffers[rank]
    symm_signals = nvshmem_create_tensors((num_ranks * COMM_SPLIT, ), NVSHMEM_SIGNAL_DTYPE,
                                          rank, LOCAL_WORLD_SIZE)
    symm_signal = symm_signals[rank]
    # Calculate golden
    a_full = torch.empty([M, K], dtype=dtype, device="cuda")

    golden = torch_ag_gemm(a_full, local_a, b, TP_GROUP)

    #####################
    # Copy Engine
    symm_signal.fill_(0)  # The initial value of signal should be 0s
    
    for _ in range(3):
        a_symm_buffer.fill_(-1)  # reset buffer
        a_symm_buffer[
            rank * M_per_rank:(rank + 1) * M_per_rank,
        ].copy_(local_a)
        # We need barrier all to make sure the above initialization visible to other ranks
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

        c = matmul_tlx_ws_persistent_ag(a_symm_buffers, b, symm_signal, num_ranks, rank, COMM_SPLIT)
        assert torch.allclose(c, golden, atol=1e-2, rtol=1e-2)
        dist_print(f"Rank {rank}", "Pass!✅", need_sync=True, allowed_ranks="all")

    quantiles = [0.5]
    warmup_time = 20
    test_time = 100

    _, ms_torch = perf_func(lambda: torch_ag_gemm(a_full, local_a, b, TP_GROUP), warmup_iters=warmup_time, iters=test_time)
    dist_print(f"Rank {rank} torch.matmul + allgather done.", need_sync=True, allowed_ranks="all")
    _, ms_triton = perf_func(lambda: matmul_tlx_ws_persistent_ag(a_symm_buffers, b, symm_signal, num_ranks, rank, COMM_SPLIT), warmup_iters=warmup_time, iters=test_time)

    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    dist_print(f"Rank {rank} torch.matmul + allgather: {perf(ms_torch):.2f} TFLOPS, tlx + nvshmem: {perf(ms_triton):.2f}", need_sync=True, allowed_ranks="all")

    nvshmem_free_tensor_sync(a_symm_buffer)
    nvshmem_free_tensor_sync(symm_signal)
    nvshmem.core.finalize()
    torch.distributed.destroy_process_group()

    # if is_cuda() and torch.cuda.get_device_capability()[0] == 9:
    #     print("Running benchmarks...")
    #     benchmark.run(show_plots=True, print_data=True, diff_col=True)
    # else:
    #     print("Skipping benchmarks, no Hopper GPU found.")
