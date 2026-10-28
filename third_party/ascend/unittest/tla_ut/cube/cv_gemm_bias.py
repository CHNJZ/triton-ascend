"""Single-kernel Cube+Vector: fp32 Y = fp16 A @ B + 1.25.

Explicit ND2NZ fills L1, MMAD produces L0C, Fixpipe moves to UB, and a
manual SIMD function adds bias before the explicit UB-to-GM store.
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
BIAS = 1.25
SIGNATURE = {"A": "*fp16", "B": "*fp16", "Y": "*fp32"}


@triton.jit
def cv_gemm_bias(A, B, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BIAS: tl.constexpr):
    tl.static_assert(M % 16 == 0 and K % 16 == 0 and N == 64)
    # Fixpipe ROW_SPLIT gives each AIV half the rows in its own UB.
    product_ub = tla.alloc((M // 2, N), tl.float32, space="ub")
    with tla.scope.cube():
        a_l1 = tla.alloc((K // 16, M // 16, 16, 16), tl.float16, space="l1")
        b_l1 = tla.alloc((N // 16, K // 16, 16, 16), tl.float16, space="l1")
        acc = tla.alloc((N // 16, M // 16, 16, 16), tl.float32, space="l0c")
        tla.nd2nz_gm_to_l1(tla.view(A, (M, K)), a_l1)
        tla.nd2nz_gm_to_l1(tla.view(B, (K, N)), b_l1)
        tla.cube.mmad(a_l1, b_l1, acc, clear_accum=True)
        tla.fixpipe_l0c_to_dst(acc,
                               product_ub,
                               dma_mode=tla.FixpipeDMAMode.NZ2ND,
                               dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                               sub_block_idx=None)
        tla.sync_block_set("CUBE", "PIPE_FIX", "PIPE_S", 0)
    with tla.scope.vector():
        tla.sync_block_wait("VECTOR", "PIPE_FIX", "PIPE_S", 0)
        with tla.scope.simd():
            bias = tla.simd.vfull(BIAS, tl.float32, 64)
            for row in tl.static_range(M // 2):
                values = tla.simd.vlds(product_ub[row, 0])
                tla.simd.vsts(product_ub[row, 0], tla.simd.vadd(values, bias))
        row_offset = tla.sub_block_id() * (M // 2)
        tla.store_ub_to_gm(product_ub, tla.view(Y, (M // 2, N), strides=(N, 1), offsets=(row_offset, 0)))


def test_ir(output_dir=None):
    module, ctx = compile_offline(cv_gemm_bias,
                                  SIGNATURE,
                                  dict(SHAPE, BIAS=BIAS),
                                  output_dir=output_dir,
                                  enable_tla_mode=True)
    text = str(lower_offline(module, ctx, output_dir=output_dir, enable_tla_mode=True))
    for op, count in (("nd2nz", 2), ("mmadL1", 1), ("fixpipe", 1), ("store", 1)):
        assert text.count("hivm.hir." + op + " ") == count, op
    assert "dma_mode = #hivm.dma_mode<nz2nd>" in text
    assert "#hivm.address_space<ub>" in text
    assert text.count("memref.alloc()") == 4
    assert "#hivm.tcore_type<CUBE>" in text
    assert text.count("hivm.vector_function") == 1
    assert "ROW_SPLIT" in text and "hivm.hir.get_sub_block_idx" in text
    assert text.count("hivm.hir.sync_block_set") == 1
    assert text.count("hivm.hir.sync_block_wait") == 1
    assert text.count("ave.hir.vadd") == SHAPE["M"] // 2
    assert 'mix_mode = "mix"' in text
    assert "hivm.module_core_type = #hivm.module_core_type<MIX>" in text
    assert "hivm.func_core_type = #hivm.func_core_type<MIX>" in text
    assert "!tt.ptr" not in text and "tt.store" not in text
    assert "tensor<" not in text and "bufferization." not in text
    print("IR PASS: cv_gemm_bias; Cube MMAD/Fixpipe -> SIMD add -> UB-to-GM")


def _launch(seed=7, compile_only=False):
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn((SHAPE["M"], SHAPE["K"]), generator=generator, dtype=torch.float16)
    b = torch.randn((SHAPE["K"], SHAPE["N"]), generator=generator, dtype=torch.float16)
    x, y = a.npu(), b.npu()
    out = torch.full((SHAPE["M"], SHAPE["N"]), float("nan"), dtype=torch.float32, device="npu")
    if compile_only:
        kernel = cv_gemm_bias.warmup(x, y, out, grid=(1, ), enable_tla_mode=True, BIAS=BIAS, **SHAPE)
        return kernel, None, None
    kernel = cv_gemm_bias[(1, )](x, y, out, enable_tla_mode=True, BIAS=BIAS, **SHAPE)
    torch.npu.synchronize()
    return kernel, out.cpu(), a.float() @ b.float() + BIAS


def test_compile():
    kernel, _, _ = _launch(compile_only=True)
    assert kernel.metadata.enable_tla_mode is True
    check_kernel_binary(kernel)


def test_run():
    for seed in (7, 19):
        _, actual, expected = _launch(seed)
        assert torch.isfinite(actual).all(), "CV left unwritten or non-finite output"
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=3e-2)
        print("ACCURACY PASS: cv_gemm_bias", SHAPE, "bias", BIAS, "seed", seed, "max_abs_error",
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
