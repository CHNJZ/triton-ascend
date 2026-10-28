"""Verify set_ctrl by reading back the bit and preserving other CTRL bits."""
import sys
from pathlib import Path

# Keep this directory off sys.path: copy.py must not shadow stdlib copy
# when any of these category files is executed directly.
_CASE_DIR = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _CASE_DIR]
sys.path.insert(0, str(_CASE_DIR.parent))

import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla
from triton.language.extra.tla.dma import _read_ctrl

from utils.tla_test_utils import compile_offline, lower_offline, get_stages, check_kernel_binary, get_tla_mode
from utils.tla_test_utils import matrix, selected

BIT = 48
OP = "set_ctrl"
CONFIGS = [dict(name="ctrl48_set_clear_restore")]


@tl.core.builtin
def _snapshot(dst, value, index, _semantic=None):
    # Test-only scalar readback: do not introduce a tensor tl.store into the
    # explicit-memref affinity pipeline just to observe the CTRL register.
    _semantic.builder.create_tla_scalar_store(value.handle, dst.handle, tl.core._unwrap_if_constexpr(index))


@triton.jit
def toggle_ctrl(Y):
    before = _read_ctrl()
    tla.set_ctrl(True, 48)
    enabled = _read_ctrl()
    tla.set_ctrl(False, 48)
    disabled = _read_ctrl()
    if (before & (1 << 48)) != 0:
        tla.set_ctrl(True, 48)
    restored = _read_ctrl()
    dst = tla.view(Y, (4, ))
    _snapshot(dst, before, 0)
    _snapshot(dst, enabled, 1)
    _snapshot(dst, disabled, 2)
    _snapshot(dst, restored, 3)


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    module, context = compile_offline(toggle_ctrl, {"Y": "*i64"}, {}, output_dir=output_dir)
    text = str(lower_offline(module, context, output_dir=output_dir))
    assert text.count("call @llvm.hivm.GET.CTRL") == 4
    assert "set_ctrl true at ctrl[48]" in text
    assert "set_ctrl false at ctrl[48]" in text
    assert text.count("memref.store ") == 4
    assert "tensor<" not in text and "bufferization." not in text
    assert "!tt.ptr" not in text and "tt.store" not in text
    print("IR PASS: ctrl")


def _launch(compile_only=False):
    output = torch.empty(4, dtype=torch.int64, device="npu")
    if compile_only:
        kernel = toggle_ctrl.warmup(output, grid=(1, ), enable_tla_mode=get_tla_mode())
        return kernel, None
    kernel = toggle_ctrl[(1, )](output, enable_tla_mode=get_tla_mode())
    torch.npu.synchronize()
    return kernel, output.cpu().tolist()


@matrix(CONFIGS)
def test_compile(case):
    kernel, _ = _launch(compile_only=True)
    check_kernel_binary(kernel)


@matrix(CONFIGS)
def test_run(case):
    _, (before, enabled, disabled, restored) = _launch()
    mask = 1 << BIT
    assert enabled & mask == mask
    assert disabled & mask == 0
    assert enabled & ~mask == before & ~mask, "set_ctrl changed unrelated bits"
    assert disabled & ~mask == before & ~mask, "set_ctrl changed unrelated bits"
    assert restored == before, "CTRL did not return to its original value"
    print("CTRL OK: bit", BIT)


def main():
    do_ir, do_compile, do_run = get_stages()
    for case in selected(CONFIGS):
        if do_ir:
            test_ir(case)
        if do_compile:
            test_compile(case)
        if do_run:
            test_run(case)
    print("ALL PASS: ctrl", "ir" if do_ir else "", "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
