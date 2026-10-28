"""Single-tile fp16 GEMM using explicit TLA DMA, with three-stage validation.

C[16, 64] = A[16, 32] @ B[32, 64], accumulated and stored as fp32.
TLA mode is always enabled. On NPU, running this file checks all three stages.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import get_stages, compile_offline, lower_offline, check_kernel_binary

SHAPE = dict(M=16, N=64, K=32)
SIGNATURE = {"A": "*fp16", "B": "*fp16", "C": "*fp32"}


@triton.jit
def gemm_explicit(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    tl.static_assert(M % 16 == 0 and N % 16 == 0 and K % 16 == 0)
    with tla.scope.cube():
        a_l1 = tla.alloc((K // 16, M // 16, 16, 16), tl.float16, space="l1")
        b_l1 = tla.alloc((N // 16, K // 16, 16, 16), tl.float16, space="l1")
        acc = tla.alloc((N // 16, M // 16, 16, 16), tl.float32, space="l0c")
        tla.nd2nz_gm_to_l1(tla.view(A, (M, K)), a_l1)
        tla.nd2nz_gm_to_l1(tla.view(B, (K, N)), b_l1)
        tla.cube.mmad(a_l1, b_l1, acc, clear_accum=True)
        tla.fixpipe_l0c_to_dst(acc, tla.view(C, (M, N)), dma_mode=tla.FixpipeDMAMode.NZ2ND)


def test_ir(output_dir=None):
    module, ctx = compile_offline(gemm_explicit, SIGNATURE, SHAPE, output_dir=output_dir, enable_tla_mode=True)
    text = str(lower_offline(module, ctx, output_dir=output_dir, enable_tla_mode=True))
    for op, count in (("nd2nz", 2), ("mmadL1", 1), ("fixpipe", 1)):
        assert text.count("hivm.hir." + op + " ") == count, op
    assert "dma_mode = #hivm.dma_mode<nz2nd>" in text
    assert text.count("memref.alloc()") == 3
    assert "#hivm.address_space<cbuf>" in text and "#hivm.address_space<cc>" in text
    assert "#hivm.tcore_type<CUBE>" in text
    assert 'mix_mode = "mix"' in text
    assert "hivm.module_core_type = #hivm.module_core_type<MIX>" in text
    assert "hivm.func_core_type = #hivm.func_core_type<MIX>" in text
    assert "hivm.vector_function" not in text
    assert "!tt.ptr" not in text and "tt.store" not in text
    assert "tensor<" not in text and "bufferization." not in text
    print("IR PASS: gemm_explicit; nd2nz x2 -> MMAD -> Fixpipe NZ2ND")


def _launch(seed=7, compile_only=False):
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn((SHAPE["M"], SHAPE["K"]), generator=generator, dtype=torch.float16)
    b = torch.randn((SHAPE["K"], SHAPE["N"]), generator=generator, dtype=torch.float16)
    x, y = a.npu(), b.npu()
    out = torch.full((SHAPE["M"], SHAPE["N"]), float("nan"), dtype=torch.float32, device="npu")
    if compile_only:
        kernel = gemm_explicit.warmup(x, y, out, grid=(1, ), enable_tla_mode=True, **SHAPE)
        return kernel, None, None
    kernel = gemm_explicit[(1, )](x, y, out, enable_tla_mode=True, **SHAPE)
    torch.npu.synchronize()
    return kernel, out.cpu(), a.float() @ b.float()


def test_compile():
    kernel, _, _ = _launch(compile_only=True)
    assert kernel.metadata.enable_tla_mode is True
    check_kernel_binary(kernel)


def test_run():
    for seed in (7, 19):
        _, actual, expected = _launch(seed)
        assert torch.isfinite(actual).all(), "GEMM left unwritten or non-finite output"
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=3e-2)
        print("ACCURACY PASS: gemm_explicit", SHAPE, "seed", seed, "max_abs_error",
              (actual - expected).abs().max().item())


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()


if __name__ == "__main__":
    main()
