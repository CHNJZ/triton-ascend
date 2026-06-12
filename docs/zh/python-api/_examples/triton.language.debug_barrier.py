@triton.jit
def debug_barrier_basic(A, B, C, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Stage 1: load data
    a = tl.load(A + offsets)

    # Insert a debug barrier to make sure all threads have finished loading
    tl.debug_barrier()

    # Stage 2: process data
    b = a * 2

    # Insert another barrier to make sure all threads have finished computing
    tl.debug_barrier()

    # Stage 3: store the result
    tl.store(C + offsets, b)