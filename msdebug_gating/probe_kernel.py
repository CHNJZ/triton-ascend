#!/usr/bin/env python3
"""
gating step 0 (recommended) — produce a GUARANTEED-VALID ttadapter dump by running
a tiny real kernel through triton-ascend, so inject_fusedloc.py has real IR to work on.

This sidesteps hand-authoring adapter IR (error-prone; see fusedloc_probe.mlir for the
self-contained fallback).

Run on the NPU box:
    TRITON_KERNEL_DUMP=1 TRITON_ALWAYS_COMPILE=1 TRITON_DUMP_DIR=./dump \
        TRITON_DISABLE_LINE_INFO=0 python3 probe_kernel.py

Then find the dump:
    find ./dump -name kernel.ttadapter.mlir   # pick the one for probe_copy_kernel
and feed it to inject_fusedloc.py.

Note: TRITON_DISABLE_LINE_INFO=0 is REQUIRED for bishengir to receive
--enable-debug-info=true (default is "true" => debug info OFF). See compiler.py:496.
"""
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def probe_copy_kernel(in_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    val = tl.load(in_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, val, mask=mask)


def main():
    n = 256
    x = torch.randn(n, dtype=torch.float32).npu()
    y = torch.empty_like(x)
    grid = (1,)
    probe_copy_kernel[grid](x, y, n, BLOCK=256)
    torch.npu.synchronize()
    assert torch.allclose(x, y), "probe kernel sanity check failed"
    print("[probe] probe_copy_kernel ran; look for kernel.ttadapter.mlir in the dump dir")


if __name__ == "__main__":
    main()
