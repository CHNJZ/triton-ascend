"""GM load/store pairs: UB padding/atomic modes and plain L1 loads."""
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
from dma.copy import route_ir, l1_ir_case, l1_compile_case, l1_run_case
from dma.fixpipe import invalid_options, _check_invalid_option

LOAD_CASES = [
    dict(name=f"{dtype}_{eviction}", dtype=dtype, eviction=eviction, pad="none", rows=2)
    for dtype in ("fp16", "fp32", "int32")
    for eviction in ("EvictNormal", "EvictFirst", "EvictLast")
]
LOAD_CASES += [
    dict(name=f"{mode}_{side}", dtype="fp32", eviction="EvictNormal", pad=mode, rows=2, left=4 if side == "left" else 0,
         right=4 if side == "right" else 0)
    for mode in ("PadNull", "PadFirstElem", "PadValue")
    for side in ("left", "right")
]
LOAD_CASES += [
    dict(name=mode + "_right_single_row", dtype="fp32", eviction="EvictFirst", pad=mode, rows=1, left=0, right=4)
    for mode in ("PadNull", "PadFirstElem", "PadValue")
]
for _case in LOAD_CASES:
    if _case["pad"] == "PadNull":
        _case["expected_diagnostic"] = True
LOAD_CASES += [
    dict(name="PadNull_no_padding", dtype="fp32", eviction="EvictNormal", pad="PadNull", rows=2, left=0, right=0)
]
STORE_CASES = [dict(name="default_" + dtype, kind=None, dtype=dtype) for dtype in ("fp16", "fp32", "int32")]
STORE_CASES += [
    dict(name=kind.name.lower(), kind=kind.name, dtype="uint32" if kind.name in ("UMAX", "UMIN") else "int32")
    for kind in tla.AtomicKind
]
STORE_CASES += [dict(name="float_" + kind.lower(), kind=kind, dtype="fp32") for kind in ("ADD", "MAX", "MIN")]
L1_CASES = [{
    'name': 'fp16_16x32_nz4d',
    'dtype': 'fp16',
    'route': 'load',
    'm': 16,
    'n': 32,
    'physical': True
}, {
    'name': 'fp16_16x16_nz4d',
    'dtype': 'fp16',
    'route': 'load',
    'm': 16,
    'n': 16,
    'physical': True
}, {
    'name': 'fp16_32x32_nz4d',
    'dtype': 'fp16',
    'route': 'load',
    'm': 32,
    'n': 32,
    'physical': True
}]
CONFIGS = [
    dict(c, name="load_" + c["name"], op="load_gm_to_ub", family="load", dependencies=["store_ub_to_gm"])
    for c in LOAD_CASES
]
CONFIGS += [
    dict(c, name="store_" + c["name"], op="store_ub_to_gm", family="store", dependencies=["load_gm_to_ub"])
    for c in STORE_CASES
]
CONFIGS += [
    dict(c, name="load_l1_" + c["name"], op="load_gm_to_l1", family="l1",
         dependencies=["copy_l1_to_ub", "store_ub_to_gm"]) for c in L1_CASES
]


@triton.jit
def padded_load(src, dst, left, right):
    tla.load_gm_to_ub(src, dst, pad_mode=tla.PadMode.PadValue, pad_value=2.5, left_padding_num=left,
                      right_padding_num=right, eviction_policy=tla.EvictionPolicy.EvictLast)


def _check_ir_padding():
    signature = {
        "src": tla.buffer_type((1, 60), tl.float16, "gm"),
        "dst": tla.buffer_type((1, 64), tl.float16, "ub"),
        "left": "i32",
        "right": "i32",
    }
    module, context = compile_offline(padded_load, signature, {})
    text = str(lower_offline(module, context))
    assert "pad_mode = <PadValue>" in text
    assert "eviction_policy = <EvictLast>" in text
    assert "pad_value" in text and "f16" in text
    assert "left_padding_num" in text and "right_padding_num" in text
    assert text.count("arith.index_cast") >= 2


def _check_unpadded_shape_mismatch():
    with pytest.raises(RuntimeError, match="TLA module verification failed"):
        route_ir("gm_ub", "gm", "ub", dst_shape=(16, 65))


def _check_backend_diagnostics():
    # These reach HIVM verification after frontend IR construction. A Python
    # argument error would raise CompilationError instead and fail the test.
    with pytest.raises(RuntimeError, match="TLA module verification failed"):
        route_ir("gm_ub", "l1", "ub")
    _check_unpadded_shape_mismatch()
    for case, src, dst in (("pad_value", "gm", "ub"), ("unit_flag_group", "l0c", "gm")):
        signature = {
            "src": tla.buffer_type((16, 64), tl.float32, src),
            "dst": tla.buffer_type((16, 64), tl.float32, dst),
        }
        with pytest.raises(RuntimeError, match="TLA module verification failed"):
            compile_offline(invalid_options, signature, {"CASE": case})


@triton.jit
def load_kernel(X, Y, left, right, DTYPE: tl.constexpr, PAD: tl.constexpr, EVICT: tl.constexpr, ROWS: tl.constexpr,
                COLS: tl.constexpr):
    # Keep room for the padded logical destination at a nonzero offset.
    # MTE writes left padding before that offset; the allocation covers both.
    storage_cols: tl.constexpr = 72 if PAD is not None and PAD != tla.PadMode.PadNull and ROWS > 1 else 64
    ub = tla.alloc((ROWS, storage_cols), DTYPE, is_mem_unique=True)
    dst = tla.view(ub, (ROWS, 64), offsets=(0, left)) if storage_cols == 72 else ub
    tla.load_gm_to_ub(tla.view(X,
                               (ROWS, COLS)), dst, pad_mode=PAD, pad_value=2.5 if PAD == tla.PadMode.PadValue else None,
                      left_padding_num=left if PAD is not None else None,
                      right_padding_num=right if PAD is not None else None, eviction_policy=EVICT)
    tla.store_ub_to_gm(tla.view(ub, (ROWS, 64)), tla.view(Y, (ROWS, 64)))


def _load_args(case):
    left, right = case.get("left", 0), case.get("right", 0)
    return dict(DTYPE={"fp16": tl.float16, "fp32": tl.float32, "int32": tl.int32}[case["dtype"]],
                PAD=None if case["pad"] == "none" else getattr(tla.PadMode, case["pad"]),
                EVICT=getattr(tla.EvictionPolicy, case["eviction"]), ROWS=case["rows"], COLS=64 - left - right)


def _load_ir(case, output_dir=None):
    if case.get("expected_diagnostic"):
        # PadNull means no padding, not zero-fill: unequal shapes must fail.
        with pytest.raises(RuntimeError, match="TLA module verification failed"):
            lowered(load_kernel, {"X": "*fp32", "Y": "*fp32", "left": "i32", "right": "i32"}, _load_args(case),
                    output_dir)
        print("EXPECTED DIAGNOSTIC PASS: PadNull rejects unequal shapes")
        return
    if case["name"] == "load_fp16_EvictNormal":
        _check_ir_padding()
        _check_backend_diagnostics()
        _check_invalid_option("enum", "gm", "ub", "expected EvictionPolicy")
    text = lowered(load_kernel, {"X": "*" + case["dtype"], "Y": "*" + case["dtype"], "left": "i32", "right": "i32"},
                   _load_args(case), output_dir)
    assert text.count("hivm.hir.load ") == 1 and text.count("hivm.hir.store ") == 1
    assert case["eviction"] in text
    if case["pad"] != "none":
        assert case["pad"] in text and "left_padding_num" in text and "right_padding_num" in text
    print("IR PASS:", case["op"], case["name"])


def _load_launch(case, compile_only=False):
    kw = _load_args(case)
    dtype = {"fp16": torch.float16, "fp32": torch.float32, "int32": torch.int32}[case["dtype"]]
    src = pattern(case["rows"], kw["COLS"], dtype)
    # Sentinel makes untouched padding distinguishable from zero/value padding.
    y = torch.full((case["rows"], 64), -999, dtype=dtype, device="npu")
    x = src.npu()
    args = (x, y, case.get("left", 0), case.get("right", 0))
    if compile_only:
        return load_kernel.warmup(*args, grid=(1, ), enable_tla_mode=get_tla_mode(), **kw), None, None
    kernel = load_kernel[(1, )](*args, enable_tla_mode=get_tla_mode(), **kw)
    torch.npu.synchronize()
    expected = torch.empty_like(y, device="cpu")
    left, right = case.get("left", 0), case.get("right", 0)
    if case["pad"] == "PadFirstElem":
        expected[:] = src[:, :1]
    else:
        expected.fill_(2.5 if case["pad"] == "PadValue" else 0)
    expected[:, left:64 - right] = src
    return kernel, y, expected


def _load_compile(case):
    if case.get("expected_diagnostic"):
        pytest.skip("PadNull denotes no padding; unequal shapes are a verifier negative test")
    check_case_binary(_load_launch(case, True)[0])


def _load_run(case):
    if case.get("expected_diagnostic"):
        pytest.skip("Invalid PadNull shape combination is not executed")
    _, actual, expected = _load_launch(case)
    try:
        exact(actual, expected)
    except AssertionError:
        print("ACTUAL_FIRST_ROW_HEAD_TAIL:", actual.cpu()[0, :8].tolist(), actual.cpu()[0, -8:].tolist())
        print("EXPECTED_FIRST_ROW_HEAD_TAIL:", expected[0, :8].tolist(), expected[0, -8:].tolist())
        raise
    print("ACCURACY PASS:", case["op"], case["name"])


@triton.jit
def store_kernel(X, Y, DTYPE: tl.constexpr, KIND: tl.constexpr):
    ub = tla.alloc((2, 64), DTYPE, is_mem_unique=True)
    tla.load_gm_to_ub(tla.view(X, (2, 64)), ub)
    tla.store_ub_to_gm(ub, tla.view(Y, (2, 64)), atomic_kind=KIND)


def _store_args(case):
    return dict(DTYPE={"fp16": tl.float16, "fp32": tl.float32, "int32": tl.int32, "uint32": tl.uint32}[case["dtype"]],
                KIND=None if case["kind"] is None else getattr(tla.AtomicKind, case["kind"]))


def _store_ir(case, output_dir=None):
    text = lowered(store_kernel, {"X": "*" + case["dtype"], "Y": "*" + case["dtype"]}, _store_args(case), output_dir)
    assert text.count("hivm.hir.store ") == 1
    if case["kind"] is not None:
        assert "atomic = <" + case["kind"].lower() + ">" in text
    print("IR PASS:", case["op"], case["name"])


def _store_inputs(case):
    dtype = {"fp16": torch.float16, "fp32": torch.float32}.get(case["dtype"], torch.int32)
    src = (torch.arange(128).reshape(2, 64) % 23 - 11).to(dtype)
    initial = (torch.arange(128).reshape(2, 64) % 19 - 9).to(dtype)
    return src, initial


def _store_launch(case, compile_only=False):
    src, initial = _store_inputs(case)
    # Negative signed storage represents values above 2**31 in uint32 mode.
    x, y = src.npu(), initial.npu()
    args = (triton.reinterpret(x, tl.uint32), triton.reinterpret(y, tl.uint32)) if case["dtype"] == "uint32" else (x, y)
    if compile_only:
        return store_kernel.warmup(*args, grid=(1, ), enable_tla_mode=get_tla_mode(), **_store_args(case)), None, None
    kernel = store_kernel[(1, )](*args, enable_tla_mode=get_tla_mode(), **_store_args(case))
    torch.npu.synchronize()
    kind = case["kind"]
    if kind in (None, "NONE"):
        expected = src
    elif kind == "ADD":
        expected = initial + src
    elif kind in ("MAX", "MIN"):
        expected = torch.maximum(initial, src) if kind == "MAX" else torch.minimum(initial, src)
    elif kind in ("AND", "OR", "XOR"):
        expected = {"AND": torch.bitwise_and, "OR": torch.bitwise_or, "XOR": torch.bitwise_xor}[kind](initial, src)
    else:
        a, b = initial.to(torch.int64) & 0xffffffff, src.to(torch.int64) & 0xffffffff
        expected = (torch.maximum(a, b) if kind == "UMAX" else torch.minimum(a, b)).to(torch.int32)
    return kernel, y, expected


def _store_compile(case):
    check_case_binary(_store_launch(case, True)[0])


def _store_run(case):
    _, actual, expected = _store_launch(case)
    try:
        exact(actual, expected)
    except AssertionError:
        src, initial = _store_inputs(case)
        actual_cpu = actual.cpu()
        print("MATCHES_PLAIN_STORE:", torch.equal(actual_cpu, src))
        print("MATCHES_ATOMIC_ADD:", torch.equal(actual_cpu, initial + src))
        for label, value in (("SOURCE", src), ("INITIAL", initial), ("ACTUAL", actual_cpu), ("EXPECTED", expected)):
            print(label, value.flatten()[:16].tolist())
        raise
    print("ACCURACY PASS:", case["op"], case["name"])


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    if case["family"] == "load":
        _load_ir(case, output_dir)
    elif case["family"] == "store":
        _store_ir(case, output_dir)
    else:
        l1_ir_case(case, output_dir)


@matrix(CONFIGS)
def test_compile(case):
    if case["family"] == "load":
        _load_compile(case)
    elif case["family"] == "store":
        _store_compile(case)
    else:
        l1_compile_case(case)


@matrix(CONFIGS)
def test_run(case):
    if case["family"] == "load":
        _load_run(case)
    elif case["family"] == "store":
        _store_run(case)
    else:
        l1_run_case(case)


def main():
    do_ir, do_compile, do_run = get_stages()
    for case in selected(CONFIGS):
        if do_ir:
            test_ir(case)
        if do_compile:
            try:
                test_compile(case)
            except pytest.skip.Exception as exc:
                print("COMPILE SKIP:", case["name"], str(exc))
        if do_run:
            try:
                test_run(case)
            except pytest.skip.Exception as exc:
                print("ACCURACY SKIP:", case["name"], str(exc))


if __name__ == "__main__":
    main()
