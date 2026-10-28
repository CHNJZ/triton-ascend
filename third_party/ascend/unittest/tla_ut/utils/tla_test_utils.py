"""Shared helpers for TLA 3-in-1 tests (ir / compile / run).

Stages are driven by env ``TLA_TEST_STAGE`` (default ``all``):
- ``ir``      : offline-compile the HIVM IR finally handed to npuir and
                assert key structure (hivm.hir.load/store, multi_buffer,
                pointer_cast, ...); no NPU needed.
- ``compile`` : on top of ``ir``, full compile and confirm .o/.npubin
                artifacts; needs Ascend driver (device or op simulator).
- ``run``     : on top of ``compile``, real-device launch + accuracy; needs NPU.

``all`` (default) auto-degrades: compile/run only when NPU is available,
otherwise ir only.
"""
import os
from pathlib import Path

import pytest
import torch

from triton.language.extra.tla.compiler import ast_to_ttir, lower_ttir
from triton._C.libtriton import ir, buffer_ir, passes
from triton._C.libtriton.ascend import ir as ascend_ir


def get_tla_mode():
    """Select the backend pipeline; default to the regular flow for validation."""
    value = os.environ.get("TLA_ENABLE_MODE", "false").strip().lower()
    if value not in ("false", "true", "0", "1"):
        raise ValueError("TLA_ENABLE_MODE must be false/true or 0/1")
    return value in ("true", "1")


class Options:
    """Default offline compile options (same as legacy gen_*.py / pytest)."""
    num_warps = 4
    num_stages = 3
    num_ctas = 1
    cluster_dims = (1, 1, 1)
    enable_fp_fusion = True
    debug = False
    sanitize_overflow = True
    arch = "Ascend910_9589"
    enable_tla_mode = get_tla_mode()
    supported_fp8_dtypes = ("fp8e5", "fp8e4nv", "fp8e4b8", "fp8e4b15", "fp8e5b16")


def npu_available():
    try:
        import torch
        import torch_npu
        return torch.npu.is_available()
    except Exception:
        return False


def get_stages():
    """Parse TLA_TEST_STAGE, return (do_ir, do_compile, do_run)."""
    stage = os.environ.get("TLA_TEST_STAGE", "all").strip().lower()
    if stage == "ir":
        return True, False, False
    if stage == "compile":
        return True, True, False
    if stage == "run":
        return True, True, True
    # all (default): full run on NPU, else degrade to ir only.
    run = npu_available()
    return True, run, run


def make_context():
    ctx = ir.context()
    ir.load_dialects(ctx)
    buffer_ir.load_dialects(ctx)
    ascend_ir.load_dialects(ctx)
    return ctx


def compile_offline(fn, signature, constexprs, output_dir=None, *, enable_tla_mode=None):
    """AST -> TTIR -> symbol_dce: front-end IR handed to npuir."""
    ctx = make_context()
    options = Options()
    if enable_tla_mode is not None:
        options.enable_tla_mode = enable_tla_mode
    module = ast_to_ttir(fn, signature, constexprs, ctx, options)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "00_ast.mlir").write_text(str(module))
    pm = ir.pass_manager(ctx)
    passes.common.add_symbol_dce(pm)
    pm.run(module, "module")
    assert module.verify()
    if output_dir is not None:
        (output_dir / "01_symbol_dce.mlir").write_text(str(module))
    return module, ctx


def lower_offline(module, ctx, output_dir=None, *, enable_tla_mode=None):
    """Run full triton-ascend chain to ttadapter boundary: no TTIR leaks to npuir."""
    mode = get_tla_mode() if enable_tla_mode is None else enable_tla_mode
    return lower_ttir(module, ctx, output_dir=output_dir, enable_tla_mode=mode)


def check_lowered_ir(text, loads, stores, buffers, extra=()):
    """Unified assert: no TTIR entities + hivm load/store/multi_buffer counts + op-specific checks."""
    for forbidden in ("!tt.ptr", "tt.ptr_to_int", "bind_buffer", "bufferization.to_tensor", "tensor<"):
        assert forbidden not in text, forbidden
    assert text.count("hivm.hir.load") == loads, "unexpected hivm.hir.load count"
    assert text.count("hivm.hir.store") == stores, "unexpected hivm.hir.store count"
    assert text.count("hivm.multi_buffer") == buffers, "unexpected hivm.multi_buffer count"
    for probe in extra:
        assert probe in text, probe


def check_compile_artifacts():
    """After launch-compile, check .npubin/.o exists in TRITON_CACHE_DIR (or default)."""
    cache_candidates = []
    cache_dir = os.environ.get("TRITON_CACHE_DIR")
    if cache_dir:
        cache_candidates.append(Path(cache_dir))
    cache_candidates.append(Path.home() / ".triton" / "cache")
    for base in cache_candidates:
        if not base.is_dir():
            continue
        artifacts = sorted(base.rglob("*.npubin")) + sorted(base.rglob("*.o"))
        if artifacts:
            return [str(p) for p in artifacts[:5]]
    return []


def check_kernel_binary(kernel):
    """Verify the binary and artifact belonging to this exact specialization."""
    binary = kernel.asm.get("npubin")
    assert isinstance(binary, bytes) and binary, "kernel did not produce an NPU binary"
    paths = [Path(path) for name, path in kernel.metadata_group.items() if name.endswith((".npubin", ".o"))]
    assert paths, "kernel metadata contains no .npubin/.o artifact"
    assert any(path.is_file() and path.read_bytes() == binary
               for path in paths), ("generated binary is missing or differs from this kernel's cache artifact")
    print("COMPILE PASS:", kernel.name, "->", ", ".join(str(path) for path in paths))


def matrix(cases):
    return pytest.mark.parametrize("case", cases, ids=[case["name"] for case in cases])


def selected(cases):
    name = os.environ.get("TLA_TEST_CASE")
    result = [case for case in cases if name is None or case["name"] == name]
    if not result:
        raise ValueError(f"unknown TLA_TEST_CASE={name!r}")
    return result


def lowered(fn, signature, constants, output_dir=None):
    aliases = {"*int8": "*i8", "*int32": "*i32", "*uint32": "*u32"}
    signature = {name: aliases.get(ty, ty) if isinstance(ty, str) else ty for name, ty in signature.items()}
    if output_dir is not None:
        output_dir = Path(output_dir)
    module, ctx = compile_offline(fn, signature, constants, output_dir=output_dir)
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    assert "!tt.ptr" not in text and "tt.store" not in text
    if get_tla_mode():
        core = "MIX" if 'mix_mode = "mix"' in text else "AIV"
        assert f"hivm.module_core_type = #hivm.module_core_type<{core}>" in text
        assert f"hivm.func_core_type = #hivm.func_core_type<{core}>" in text
    return text


def exact(actual, expected):
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)


def check_case_binary(kernel):
    assert kernel.metadata.enable_tla_mode == get_tla_mode()
    check_kernel_binary(kernel)


def pattern(rows, cols, dtype):
    return (((torch.arange(rows * cols).reshape(rows, cols) * 7) % 43) - 21).to(dtype)


def pack_nz(nd):
    """L1 NZ is [N/C0, M/16, 16, C0]; half/bfloat16 use C0=16."""
    m, n = nd.shape
    c0 = 32 // nd.element_size()
    return nd.reshape(m // 16, 16, n // c0, c0).permute(2, 0, 1, 3).contiguous()
