"""Generic copy and explicit UB/L1/L0/BiasBUF/FixBUF transfer routes."""
import sys
from pathlib import Path

# Keep this directory off sys.path: copy.py must not shadow stdlib copy
# when any of these category files is executed directly.
_CASE_DIR = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _CASE_DIR]
sys.path.insert(0, str(_CASE_DIR.parent))
import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import (matrix, selected, get_stages, lowered, pattern, exact, pack_nz, get_tla_mode,
                                  check_case_binary, compile_offline, lower_offline)
from triton.compiler.errors import CompilationError

GENERIC_CASES = [dict(name="gm_ub_gm_" + dtype, dtype=dtype) for dtype in ("fp16", "fp32", "int32")]
CONFIGS = [
    dict(c, family="generic", op="copy", dependencies=["load_gm_to_ub", "store_ub_to_gm"]) for c in GENERIC_CASES
]

ROUTE_CASES = [{
    'name': 'fp16_16x16_logical2d', 'dtype': 'fp16', 'route': 'ub_to_l1', 'm': 16, 'n': 16, 'physical': False
}, {'name': 'fp16_16x16_nz4d', 'dtype': 'fp16', 'route': 'ub_to_l1', 'm': 16, 'n': 16, 'physical': True},
               {'name': 'fp16_32x32_nz4d', 'dtype': 'fp16', 'route': 'ub_to_l1', 'm': 32, 'n': 32, 'physical': True}]
CONFIGS += [
    dict(c, name="ub_to_l1_" + c["name"], family="ub_to_l1", op="copy_ub_to_l1",
         dependencies=["load_gm_to_ub", "copy_l1_to_ub", "store_ub_to_gm"]) for c in ROUTE_CASES
]

ROUTE_CASES = [{'name': 'fp16_16x16_logical2d', 'dtype': 'fp16', 'route': 'load', 'm': 16, 'n': 16, 'physical': False},
               {'name': 'fp16_16x16_nz4d', 'dtype': 'fp16', 'route': 'load', 'm': 16, 'n': 16, 'physical': True},
               {'name': 'fp16_32x32_nz4d', 'dtype': 'fp16', 'route': 'load', 'm': 32, 'n': 32, 'physical': True},
               {'name': 'fp32_16x16_nz4d', 'dtype': 'fp32', 'route': 'load', 'm': 16, 'n': 16, 'physical': True}]
CONFIGS += [
    dict(c, name="l12ub_" + c["name"], family="l12ub", op="copy_l1_to_ub",
         dependencies=["load_gm_to_l1", "store_ub_to_gm"]) for c in ROUTE_CASES
]

ROUTE_CASES = [
    dict(name=target + ("_transpose" if transpose else "_normal"), target=target, transpose=transpose,
         compile_kind="allocation_diagnostic") for target in ("l0a", "l0b") for transpose in (False, True)
]
CONFIGS += [
    dict(c, name="l12l0_" + c["name"], family="l12l0", op="copy_l1_to_l0", dependencies=["load_gm_to_l1"])
    for c in ROUTE_CASES
]

ROUTE_CASES = [
    dict(name=dtype, dtype=dtype, target="bt", compile_kind="allocation_diagnostic") for dtype in ("fp16", "fp32")
]
CONFIGS += [
    dict(c, name="l12bt_" + c["name"], family="l12bt", op="copy_l1_to_bt", dependencies=["load_gm_to_l1"])
    for c in ROUTE_CASES
]

ROUTE_CASES = [dict(name="unsupported", expected_diagnostic=True)]
CONFIGS += [
    dict(c, name="l12fb_" + c["name"], family="l12fb", op="copy_l1_to_fb", dependencies=[]) for c in ROUTE_CASES
]


@triton.jit
def route_copy(src, dst, ROUTE: tl.constexpr):
    if ROUTE == "ub_l1":
        tla.copy_ub_to_l1(src, dst)
    elif ROUTE == "l1_ub":
        tla.copy_l1_to_ub(src, dst)
    elif ROUTE == "l1_bt":
        tla.copy_l1_to_bt(src, dst)
    elif ROUTE == "l1_fb":
        tla.copy_l1_to_fb(src, dst)
    elif ROUTE == "gm_l1":
        tla.load_gm_to_l1(src, dst)
    elif ROUTE == "nd2nz":
        tla.nd2nz_gm_to_l1(src, dst, l2_cache_ctrl=3)
    elif ROUTE == "gm_ub":
        tla.load_gm_to_ub(src, dst)
    elif ROUTE == "ub_gm":
        tla.store_ub_to_gm(src, dst, atomic_kind=tla.AtomicKind.ADD)
    elif ROUTE == "l1_l0":
        tla.copy_l1_to_l0(src, dst, is_transpose=True, k_part_idx=0, k_part=64, k_part_ceil=64, k_part_loop=1,
                          k_part_actual=64, m=16)
    elif ROUTE == "actual_batch":
        tla.copy_l1_to_l0(src, dst, actual_batch=1)
    elif ROUTE == "legacy":
        tla.copy(src, dst)


def route_ir(route, src_space, dst_space, dtype=tl.float32, src_shape=(16, 64), dst_shape=None):
    signature = {
        "src": tla.buffer_type(src_shape, dtype, src_space),
        "dst": tla.buffer_type(dst_shape or src_shape, dtype, dst_space),
    }
    module, context = compile_offline(route_copy, signature, {"ROUTE": route})
    return str(lower_offline(module, context))


def _check_ir_route(route, src_space, dst_space, op):
    text = route_ir(route, src_space, dst_space)
    assert f"hivm.hir.{op} " in text
    assert "tensor<" not in text
    if route == "nd2nz":
        assert "l2_cache_mode = 3 : i32" in text
    if route == "ub_gm":
        assert "atomic" in text and "add" in text
    if route == "l1_l0":
        assert "is_transposed = true" in text
        assert "k_part_actual = 64" in text
    if route == "l1_bt":
        assert "#hivm.address_space<biasbuf>" in text
    if route in ("nd2nz", "l1_l0", "l1_ub", "l1_bt") or (route == "legacy" and src_space == "l1"):
        assert 'mix_mode = "mix"' in text


def _check_invalid_route(route, src, dst, message):
    with pytest.raises(CompilationError, match=message):
        route_ir(route, src, dst)


@triton.jit
def _l1_roundtrip(X, Y, ROUTE: tl.constexpr, DTYPE: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
                  PHYSICAL: tl.constexpr, CACHE: tl.constexpr):
    c0: tl.constexpr = 8 if DTYPE == tl.float32 else 16
    shape: tl.constexpr = (N // c0, M // 16, 16, c0) if PHYSICAL else (M, N)
    l1 = tla.alloc(shape, DTYPE, space="l1", is_mem_unique=True)
    out = tla.alloc((M, N), DTYPE, is_mem_unique=True)
    if ROUTE == "nd2nz":
        with tla.scope.cube():
            tla.nd2nz_gm_to_l1(tla.view(X, (M, N)), l1, l2_cache_ctrl=CACHE)
    elif ROUTE == "ub_to_l1":
        ub = tla.alloc(shape, DTYPE, is_mem_unique=True)
        tla.load_gm_to_ub(tla.view(X, shape), ub)
        tla.copy_ub_to_l1(ub, l1)
    else:
        with tla.scope.cube():
            tla.load_gm_to_l1(tla.view(X, shape), l1)
    with tla.scope.cube():
        tla.copy_l1_to_ub(l1, out)
    tla.store_ub_to_gm(out, tla.view(Y, (M, N)))


def _l1_args(case):
    return dict(ROUTE=case["route"], DTYPE={"fp16": tl.float16, "fp32": tl.float32}[case["dtype"]], M=case["m"],
                N=case["n"], PHYSICAL=case["physical"], CACHE=case.get("cache"))


def l1_ir_case(case, output_dir=None):
    text = lowered(_l1_roundtrip, {"X": "*" + case["dtype"], "Y": "*" + case["dtype"]}, _l1_args(case), output_dir)
    assert text.count("hivm.hir.l12ub ") == 1 and text.count("hivm.hir.store ") == 1
    op = {"nd2nz": "nd2nz", "ub_to_l1": "copy", "load": "load"}[case["route"]]
    assert "hivm.hir." + op + " " in text
    if case.get("cache") is not None:
        assert f"l2_cache_mode = {case['cache']} : i32" in text
    if case["physical"]:
        assert f"memref<{case['n'] // (8 if case['dtype'] == 'fp32' else 16)}x{case['m'] // 16}x16x" in text
    print("IR PASS:", case["name"])


def _l1_launch(case, compile_only=False):
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[case["dtype"]]
    expected = pattern(case["m"], case["n"], dtype)
    # Plain load/copy must be given prepacked NZ bytes. ND2NZ gets row-major ND.
    src = pack_nz(expected) if case["physical"] and case["route"] != "nd2nz" else expected
    x = src.npu()
    y = torch.full_like(expected, float("nan"), device="npu")
    if compile_only:
        return _l1_roundtrip.warmup(x, y, grid=(1, ), enable_tla_mode=get_tla_mode(), **_l1_args(case)), None, None
    kernel = _l1_roundtrip[(1, )](x, y, enable_tla_mode=get_tla_mode(), **_l1_args(case))
    torch.npu.synchronize()
    return kernel, y, expected


def l1_compile_case(case):
    check_case_binary(_l1_launch(case, True)[0])


def l1_run_case(case):
    _, actual, expected = _l1_launch(case)
    exact(actual, expected)
    print("ACCURACY PASS:", case["name"])


@triton.jit
def generic_copy(X, Y, DTYPE: tl.constexpr):
    ub = tla.alloc((2, 64), DTYPE, is_mem_unique=True)
    tla.copy(tla.view(X, (2, 64)), ub)
    tla.copy(ub, tla.view(Y, (2, 64)))


def _generic_dtype(case):
    return {"fp16": tl.float16, "fp32": tl.float32, "int32": tl.int32}[case["dtype"]]


def _generic_ir(case, output_dir=None):
    _check_ir_route("legacy", "ub", "l1", "copy")
    _check_ir_route("legacy", "l1", "ub", "l12ub")
    text = lowered(generic_copy, {"X": "*" + case["dtype"], "Y": "*" + case["dtype"]}, {"DTYPE": _generic_dtype(case)},
                   output_dir)
    assert text.count("hivm.hir.load ") == 1 and text.count("hivm.hir.store ") == 1
    print("IR PASS: generic copy", case["name"])


def _generic_launch(case, compile_only=False):
    dtype = {"fp16": torch.float16, "fp32": torch.float32, "int32": torch.int32}[case["dtype"]]
    expected = (torch.arange(128).reshape(2, 64) - 57).to(dtype)
    x, y = expected.npu(), torch.full_like(expected, -999, device="npu")
    if compile_only:
        return generic_copy.warmup(x, y, DTYPE=_generic_dtype(case), grid=(1, ),
                                   enable_tla_mode=get_tla_mode()), None, None
    kernel = generic_copy[(1, )](x, y, DTYPE=_generic_dtype(case), enable_tla_mode=get_tla_mode())
    torch.npu.synchronize()
    return kernel, y, expected


def _generic_compile(case):
    check_case_binary(_generic_launch(case, True)[0])


def _generic_run(case):
    _, actual, expected = _generic_launch(case)
    exact(actual, expected)
    print("ACCURACY PASS: generic copy", case["name"])


@triton.jit
def native_copy(src, dst, TARGET: tl.constexpr, TRANSPOSE: tl.constexpr):
    if TARGET == "bt":
        tla.copy_l1_to_bt(src, dst)
    else:
        tla.copy_l1_to_l0(src, dst, is_transpose=TRANSPOSE, k_part_idx=0, k_part=64, k_part_ceil=64, k_part_loop=1,
                          k_part_actual=64, m=16)


def _native_ir(case, output_dir=None):
    dtype = tl.float32 if case.get("dtype") == "fp32" else tl.float16
    text = lowered(
        native_copy,
        {"src": tla.buffer_type((16, 64), dtype, "l1"), "dst": tla.buffer_type((16, 64), dtype, case["target"])},
        dict(TARGET=case["target"], TRANSPOSE=case.get("transpose", False)), output_dir)
    assert "hivm.hir." + ("l12bt" if case["target"] == "bt" else "l12l0") + " " in text
    if case["target"] != "bt":
        assert "k_part_actual = 64" in text
        if case["transpose"]:
            assert "is_transposed = true" in text
        else:
            assert "is_transposed = true" not in text
    else:
        assert "n = %c64" in text and "arith.constant 64 : index" in text and "biasbuf" in text
    print("IR PASS:", case["name"])


@triton.jit
def allocation_probe(X, Y, TARGET: tl.constexpr, TRANSPOSE: tl.constexpr, DTYPE: tl.constexpr):
    src = tla.alloc((16, 64), DTYPE, space="l1", is_mem_unique=True)
    dst = tla.alloc((16, 64), DTYPE, space=TARGET, is_mem_unique=True)
    with tla.scope.cube():
        tla.load_gm_to_l1(tla.view(X, (16, 64)), src)
        native_copy(src, dst, TARGET, TRANSPOSE)
    tl.store(Y, 1)


def _native_compile(case):
    dtype = torch.float32 if case.get("dtype") == "fp32" else torch.float16
    x = torch.ones((16, 64), dtype=dtype, device="npu")
    y = torch.empty(1, dtype=torch.int32, device="npu")
    kernel = allocation_probe.warmup(x, y, TARGET=case["target"], TRANSPOSE=case.get("transpose", False),
                                     DTYPE=tl.float32 if dtype == torch.float32 else tl.float16, grid=(1, ),
                                     enable_tla_mode=get_tla_mode())
    check_case_binary(kernel)
    # A successful constant-store kernel is insufficient evidence: reject DCE.
    name = "l12bt" if case["target"] == "bt" else "l12l0"
    if name not in kernel.asm.get("llir", ""):
        pytest.skip("Allocation diagnostic produced no inspectable retained " + name + " LLVM operation")


def _native_run(case):
    pytest.skip("No public L0A/L0B/BiasBUF consumer; allocation probe is never launched or counted as precision PASS")


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    family = case["family"]
    if family == "generic":
        _generic_ir(case, output_dir)
    elif family in ("ub_to_l1", "l12ub"):
        l1_ir_case(case, output_dir)
    elif family in ("l12l0", "l12bt"):
        _native_ir(case, output_dir)
        if family == "l12l0":
            _check_invalid_route("actual_batch", "l1", "l0a", "actual_batch has no field")
    else:
        _check_invalid_route("l1_fb", "l1", "fb", "has no L1->FixBUF DMA operation")
        print("EXPECTED DIAGNOSTIC PASS: L1->FixBUF remains unsupported")


@matrix(CONFIGS)
def test_compile(case):
    family = case["family"]
    if family == "generic":
        _generic_compile(case)
    elif family in ("ub_to_l1", "l12ub"):
        l1_compile_case(case)
    elif family in ("l12l0", "l12bt"):
        _native_compile(case)
    else:
        pytest.skip("copy_l1_to_fb has no mapped HIVM operation")


@matrix(CONFIGS)
def test_run(case):
    family = case["family"]
    if family == "generic":
        _generic_run(case)
    elif family in ("ub_to_l1", "l12ub"):
        l1_run_case(case)
    elif family in ("l12l0", "l12bt"):
        _native_run(case)
    else:
        pytest.skip("copy_l1_to_fb has no mapped HIVM operation")


def main():
    do_ir, do_compile, do_run = get_stages()
    for case in selected(CONFIGS):
        if do_ir:
            test_ir(case)
        if do_compile:
            test_compile(case)
        if do_run:
            test_run(case)


if __name__ == "__main__":
    main()
