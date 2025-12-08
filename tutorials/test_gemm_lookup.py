from typing import Optional
import triton
import triton.language as tl
import torch

################### triton kernel ###################
@triton.jit
def kernel_gemm_rs_producer_persistent(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    local_world_size,
    pid_lookup_ptr,
    pid_max_steps,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """
    The 'kernel_gemm_rs_producer_persistent' kernel is almost identical to a regular Triton GEMM kernel, with only two minor differences:
    1. The computation order of tiles is swizzled according to the rank.
    2. There is an additional operation to set the barrier in the epilogue.
    """
    rank = 0
    num_ranks = 1
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

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
        block_shape=[
            BLOCK_SIZE_M,
            BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2,
        ],
    )

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1
    pid_max_steps_tensor = tl.full((), pid_max_steps, dtype=tl.int32)

    tile_id = start_pid - NUM_SMS
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    tile_step = tl.full((), -1, dtype=tl.int32)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_step += 1
            tile_id += NUM_SMS

            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            # lookup_index = start_pid * pid_max_steps_tensor + tile_step
            # lookup_offsets = lookup_index * 2 + tl.arange(0, 2)
            # expected = tl.load(pid_lookup_ptr + lookup_offsets)
            # expected_m = expected[0]
            # expected_n = expected[1]
            # tl.device_assert(expected_m >= 0, "tile lookup accessed padding")
            # tl.device_assert((expected_m == pid_m) & (expected_n == pid_n),
            #                  "pid mismatch with lookup")

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

        offs_k = ki * BLOCK_SIZE_K

        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            if EPILOGUE_SUBTILE:
                acc = tl.reshape(accumulator,
                                 (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
                acc = tl.permute(acc, (0, 2, 1))
                acc0, acc1 = tl.split(acc)
                c0 = acc0.to(dtype)
                c_desc.store([offs_am, offs_bn], c0)
                c1 = acc1.to(dtype)
                c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1)
            else:
                c = accumulator.to(dtype)
                c_desc.store([offs_am, offs_bn], c)
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N),
                                   dtype=tl.float32)


def gemm_rs_producer_persistent(a,
                                b,
                                c,
                                world_size,
                                local_world_size,
                                num_gemm_sms,
                                BLOCK_SIZE_M=128,
                                BLOCK_SIZE_N=256,
                                BLOCK_SIZE_K=64,
                                GROUP_SIZE_M=8,
                                STAGES=3):
    # Check constraints.
    assert a.shape[1] == b.shape[
        1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, local_K = a.shape
    N, local_K = b.shape

    M_per_rank = M // world_size

    assert M_per_rank % BLOCK_SIZE_M == 0

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    pid_lookup = get_tile_order(M,
                                N,
                                local_K,
                                BLOCK_SIZE_M,
                                BLOCK_SIZE_N,
                                BLOCK_SIZE_K,
                                num_gemm_sms,
                                group_size_m=GROUP_SIZE_M,
                                device=a.device).contiguous()
    pid_max_steps = pid_lookup.shape[1]
    pid_lookup_flat = pid_lookup.view(-1)

    grid = lambda META: (min(
        num_gemm_sms,
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(
            N, META["BLOCK_SIZE_N"]),
    ), )

    # Launch the Triton GEMM kernel. Once the kernel has completed the computation of the output tiles
    # that send to a specific rank, will set the corresponding barrier to 1.
    compiled = kernel_gemm_rs_producer_persistent[grid](
        a,
        b,
        c,
        M,
        N,
        local_K,
        local_world_size,
        pid_lookup_flat,
        pid_max_steps,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
        False,
        NUM_SMS=num_gemm_sms,  #
        num_stages=STAGES,
        num_warps=8,
    )

    return compiled

def get_tile_order(M,
                   N,
                   K,
                   BLOCK_SIZE_M,
                   BLOCK_SIZE_N,
                   BLOCK_SIZE_K,
                   num_sms,
                   group_size_m=8,
                   device=None):
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    num_pid_m = triton.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n
    if num_tiles == 0 or num_sms == 0:
        return torch.empty((num_sms, 0, 2), dtype=torch.int32, device=device)
    tiles_per_sm = [num_tiles // num_sms for _ in range(num_sms)]
    for i in range(num_tiles % num_sms):
        tiles_per_sm[i] += 1
    max_steps = max(tiles_per_sm)
    pid_tensor = torch.full((num_sms, max_steps, 2),
                            -1,
                            dtype=torch.int32,
                            device=device)
    num_pid_in_group = group_size_m * num_pid_n
    for sm_id in range(num_sms):
        for step in range(tiles_per_sm[sm_id]):
            tile_id = sm_id + step * num_sms
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * group_size_m
            group_size = min(num_pid_m - first_pid_m, group_size_m)
            intra_group_idx = tile_id % num_pid_in_group
            pid_m = first_pid_m + (intra_group_idx % group_size)
            pid_n = intra_group_idx // group_size
            pid_tensor[sm_id, step, 0] = pid_m
            pid_tensor[sm_id, step, 1] = pid_n
    return pid_tensor.permute(1, 0, 2).contiguous()

if __name__ == "__main__":
    M = 8192
    N = 8192
    K = 8192
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((N, K), device='cuda', dtype=torch.float16)
    c = torch.zeros((M, N), device='cuda', dtype=torch.float16)

    gemm_rs_producer_persistent(
        a,
        b,
        c,
        1,  # world_size
        1,  # local_world_size
        132,  # num_gemm_sms
    )

    golden = torch.matmul(a, b.T)
    torch.testing.assert_close(c, golden, rtol=1e-2, atol=1e-2)

    ms = triton.testing.do_bench(
        lambda: torch.matmul(a, b.T),
        rep=2000,
        warmup=200,
    )

    ms_triton = triton.testing.do_bench(
        lambda: gemm_rs_producer_persistent(
            a,
            b,
            c,
            1,  # world_size
            1,  # local_world_size
            132,  # num_gemm_sms
        ),
        rep=2000,
        warmup=200,
    )

    cal_tflops = lambda ms: 2 * M * N * K / (ms / 1000) / 1e12
    print(
        f"gemm_rs_producer_persistent: {ms_triton:.2f}ms, {cal_tflops(ms_triton):.2f} TFlops, speedup {ms/ms_triton:.2f}x over torch.matmul {ms:.2f}ms, {cal_tflops(ms):.2f} TFlops"
    )
