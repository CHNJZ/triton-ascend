#!/usr/bin/env python3
"""Probe richer compute to find FIXABLE line losses: ops that DO lower to real
instructions but get mis-tagged (fusion/CSE) — unlike constant/view absorptions.
Each compute line below is a real elementwise instruction.

Run:
  TRITON_KERNEL_DUMP=1 TRITON_ALWAYS_COMPILE=1 TRITON_DUMP_DIR=./cmp_dump \
    TRITON_DISABLE_LINE_INFO=0 LLVM_EXTRACT_DI_LOCAL_VARIABLES=1 python3 probe_compute.py
"""
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def compute_kernel(in_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                      # 19 orphan
    offs = pid * BLOCK + tl.arange(0, BLOCK)    # 20 index (dynamic)
    mask = offs < n                             # 21 mask
    x = tl.load(in_ptr + offs, mask=mask)       # 22 load
    a = x * 2.0                                 # 23 compute
    b = a + 1.0                                 # 24 compute
    c = tl.sqrt(b)                              # 25 compute
    d = c - x                                   # 26 compute
    tl.store(out_ptr + offs, d, mask=mask)      # 27 store


def main():
    n = 256
    x = torch.rand(n, dtype=torch.float32).npu() + 1.0
    y = torch.empty_like(x)
    compute_kernel[(1,)](x, y, n, BLOCK=256)
    torch.npu.synchronize()
    ref = torch.sqrt(x * 2.0 + 1.0) - x
    assert torch.allclose(y, ref, atol=1e-3), "compute probe sanity check failed"
    print("[probe] compute_kernel ran")


if __name__ == "__main__":
    main()
