@triton.jit
def basic_assume_example(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    # Assume BLOCK_SIZE is a power of 2 so the compiler can optimize the division
    tl.assume((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0)

    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)

    # Knowing BLOCK_SIZE is a power of 2, the compiler can lower the division into a shift
    result = x // BLOCK_SIZE + y % BLOCK_SIZE
    tl.store(y_ptr + offsets, result)