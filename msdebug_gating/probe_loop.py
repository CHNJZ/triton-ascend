import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def loop_kernel(in_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in range(0, 4):
        m = offs < n
        v = tl.load(in_ptr + k * BLOCK + offs, mask=m)
        acc = acc + v
    tl.store(out_ptr + offs, acc, mask=offs < n)


def main():
    n = 256
    x = torch.rand(4 * n, dtype=torch.float32).npu()
    y = torch.empty(n, dtype=torch.float32).npu()
    loop_kernel[(1,)](x, y, n, BLOCK=256)
    torch.npu.synchronize()
    print("[probe] loop_kernel ran")


if __name__ == "__main__":
    main()
