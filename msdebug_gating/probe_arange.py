#!/usr/bin/env python3
"""A2 probe: pure-arange index so the index line gets absorbed into the
reinterpret_cast offset with no surviving scalar op (unlike probe_kernel.py where
`offs = pid*BLOCK + arange` survives via the muli). Used to exercise the index/arange
loc-preservation (task #5).

Run:
  TRITON_KERNEL_DUMP=1 TRITON_ALWAYS_COMPILE=1 TRITON_DUMP_DIR=./a2_dump \
    TRITON_DISABLE_LINE_INFO=0 LLVM_EXTRACT_DI_LOCAL_VARIABLES=1 python3 probe_arange.py
"""
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def arange_copy_kernel(in_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)                 # line 21 — pure arange index
    mask = offs < n                            # line 22 — mask
    val = tl.load(in_ptr + offs, mask=mask)    # line 23 — load
    tl.store(out_ptr + offs, val, mask=mask)   # line 24 — store


def main():
    n = 256
    x = torch.randn(n, dtype=torch.float32).npu()
    y = torch.empty_like(x)
    arange_copy_kernel[(1,)](x, y, n, BLOCK=256)
    torch.npu.synchronize()
    assert torch.allclose(x, y), "arange probe sanity check failed"
    print("[probe] arange_copy_kernel ran")


if __name__ == "__main__":
    main()
