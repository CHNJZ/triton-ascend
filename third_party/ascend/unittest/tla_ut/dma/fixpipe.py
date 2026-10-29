"""Fixpipe modes and paired MMAD producer; record dependency failures separately."""
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

OP = "fixpipe_l0c_to_dst"
DEPENDENCIES = ["nd2nz_gm_to_l1", "mmad", "store_ub_to_gm"]


def config(name, **kw):
    values = dict(name=name,
                  out="fp32",
                  mode="NZ2ND",
                  relu="NO_RELU",
                  unit=None,
                  accumulate=False,
                  dest="gm",
                  quant="NO_QUANT",
                  scale=None,
                  dual="NO_DUAL",
                  sub="SUB_BLOCK_0",
                  channel=False,
                  c0=True,
                  transpose=False)
    values.update(kw)
    return values


CONFIGS = [
    config("gm_f32"),
    config("gm_f16", out="fp16", quant="F322F16"),
    config("gm_bf16", out="bf16", quant="F322BF16"),
    config("gm_nz2nz", out="fp16", mode="NZ2NZ", quant="F322F16"),
    config("gm_nz2dn", mode="NZ2DN"),
    config("gm_relu", relu="NORMAL_RELU"),
    config("gm_scaled", quant="QF322F32_PRE", scale=0.5),
    config("gm_unit0", unit=0),
    config("gm_unit3", unit=3),
    config("gm_unit2_then3", unit=3, accumulate=True),
    config("ub_f32", dest="ub"),
    config("subblock1", sub="SUB_BLOCK_1"),
    config("c0_pad_off", c0=False),
    config("channel_split", channel=True)
]
CONFIGS += [
    config("dual_" + mode.lower(), dest="ub", out="fp16", quant="F322F16", dual=mode, sub=None,
           accuracy_reason="Split UB output needs a two-subblock address/reference protocol")
    for mode in ("ROW_SPLIT", "COLUMN_SPLIT")
]
CONFIGS += [
    config("relu_" + mode.lower(), relu=mode,
           accuracy_reason="The API has no slope/PReLU parameter; check IR and binary only")
    for mode in ("LEAKY_RELU", "P_RELU")
]
CONFIGS += [
    config("s32_to_i8", out="int8", quant="S322I8",
           accuracy_reason="S322I8 scale/rounding protocol has no confirmed CPU oracle")
]
for _case in CONFIGS:
    _case["op"] = "fixpipe_l0c_to_dst"
    _case["dependencies"] = ["nd2nz_gm_to_l1", "mmad", "store_ub_to_gm"]
CONFIGS += [
    dict(c, name="mmad_" + c["name"], op="mmad", dependencies=["nd2nz_gm_to_l1", "fixpipe_l0c_to_dst"])
    for c in (config("clear"), config("transpose_b", transpose=True), config("unit0", unit=0),
              config("unit3", unit=3), config("accumulate_unit2_then3", unit=3, accumulate=True))
]


@triton.jit
def fixpipe_options(src, dst, scale):
    tla.fixpipe_l0c_to_dst(src, dst, dma_mode=tla.FixpipeDMAMode.NZ2ND, sub_block_idx=tla.FixpipeSubBlock.SUB_BLOCK_1,
                           pre_quant=tla.FixpipePreQuantMode.QF322F32_PRE, pre_relu=tla.FixpipePreReluMode.NORMAL_RELU,
                           c0_pad_en=False, quant_scale=scale)
    tla.set_ctrl(True, 48)
    tla.set_ctrl(False, 60)


def _check_ir_fixpipe():
    signature = {
        "src": tla.buffer_type((16, 64), tl.float32, "l0c"),
        "dst": tla.buffer_type((16, 64), tl.float32, "gm"),
        "scale": "fp32",
    }
    module, context = compile_offline(fixpipe_options, signature, {})
    text = str(lower_offline(module, context))
    for expected in ("nz2nd", "sub_block_1", "QF322F32_PRE", "NORMAL_RELU", "c0_pad_en = false", "quant_scale",
                     "set_ctrl true at ctrl[48]", "set_ctrl false at ctrl[60]"):
        assert expected in text
    assert 'mix_mode = "mix"' in text


@triton.jit
def invalid_options(src, dst, CASE: tl.constexpr):
    if CASE == "pad_value":
        tla.load_gm_to_ub(src, dst, pad_mode=tla.PadMode.PadValue)
    elif CASE == "enum":
        tla.load_gm_to_ub(src, dst, eviction_policy="EvictLast")
    elif CASE == "unit_flag":
        tla.fixpipe_l0c_to_dst(src, dst, unit_flag_ctrl=4)
    elif CASE == "unit_flag_bool":
        tla.fixpipe_l0c_to_dst(src, dst, unit_flag_ctrl=True)
    elif CASE == "unit_flag_group":
        tla.fixpipe_l0c_to_dst(src, dst, unit_flag_group_id=-1)


def _check_invalid_option(case, src, dst, message):
    signature = {"src": tla.buffer_type((16, 64), tl.float32, src), "dst": tla.buffer_type((16, 64), tl.float32, dst)}
    with pytest.raises(CompilationError, match=message):
        compile_offline(invalid_options, signature, {"CASE": case})


@triton.jit
def fixpipe_split(src, dst):
    tla.fixpipe_l0c_to_dst(src, dst, dma_mode=tla.FixpipeDMAMode.NZ2ND, dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                           sub_block_idx=None, pre_quant=tla.FixpipePreQuantMode.F322F16, channel_split=True)


def _check_ir_fixpipe_split_and_convert():
    signature = {"src": tla.buffer_type((16, 64), tl.float32, "l0c"), "dst": tla.buffer_type((8, 64), tl.float16, "ub")}
    module, context = compile_offline(fixpipe_split, signature, {})
    text = str(lower_offline(module, context))
    assert "ROW_SPLIT" in text and "F322F16" in text
    assert "channel_split = true" in text
    assert "memref<8x64xf16" in text


@triton.jit
def unit_flag_pair(a, b, acc, out, MODE: tl.constexpr, GROUP: tl.constexpr):
    with tla.scope.cube():
        tla.cube.mmad(a, b, acc, clear_accum=True, unit_flag_ctrl=MODE, unit_flag_group_id=GROUP)
        tla.fixpipe_l0c_to_dst(acc, out, dma_mode=tla.FixpipeDMAMode.NZ2ND, unit_flag_ctrl=MODE,
                               unit_flag_group_id=GROUP)


def _check_ir_unit_flags():
    signature = {
        "a": tla.buffer_type((16, 16), tl.float16, "l1"),
        "b": tla.buffer_type((16, 16), tl.float16, "l1"),
        "acc": tla.buffer_type((16, 16), tl.float32, "l0c"),
        "out": tla.buffer_type((16, 16), tl.float32, "gm"),
    }
    # RESERVED is representable in the native enum; it is only checked in IR,
    # never executed as a device synchronization protocol.
    for mode, group in ((None, None), (0, None), (1, None), (2, 3), (3, 3), (tla.UnitFlagMode.ENABLED_WITH_UPDATE, 0)):
        module, context = compile_offline(unit_flag_pair, signature, {"MODE": mode, "GROUP": group})
        text = str(lower_offline(module, context))
        if mode is None:
            assert "unit_flag_mode" not in text
        else:
            name = tla.UnitFlagMode(mode).name
            assert text.count(f"#hivm.unit_flag<{name}>") == 2
        if group is not None:
            assert text.count(f"unit_flag_group_id = {group} : i64") == 2
        assert "unit_flag_cond" not in text


@triton.jit
def product(A, B, Y, OUT: tl.constexpr, DMA: tl.constexpr, RELU: tl.constexpr, UNIT: tl.constexpr,
            ACCUMULATE: tl.constexpr, DEST: tl.constexpr, QUANT: tl.constexpr, SCALE: tl.constexpr,
            DUAL: tl.constexpr, SUB: tl.constexpr, CHANNEL: tl.constexpr, C0: tl.constexpr,
            TRANSPOSE: tl.constexpr):
    input_dtype: tl.constexpr = tl.int8 if OUT == "int8" else tl.float16
    acc_dtype: tl.constexpr = tl.int32 if OUT == "int8" else tl.float32
    k: tl.constexpr = 32 if OUT == "int8" else 16
    n: tl.constexpr = 32 if OUT == "int8" else 16
    c0: tl.constexpr = 32 if OUT == "int8" else 16
    dtype: tl.constexpr = tl.int8 if OUT == "int8" else (tl.float16 if OUT == "fp16" else
                                                         (tl.bfloat16 if OUT == "bf16" else tl.float32))
    rows: tl.constexpr = 8 if DUAL == tla.FixpipeDualDstMode.ROW_SPLIT else 16
    cols: tl.constexpr = n // 2 if DUAL == tla.FixpipeDualDstMode.COLUMN_SPLIT else n
    if DEST == "ub":
        dst = tla.alloc((rows, cols), dtype, is_mem_unique=True)
    elif DMA == tla.FixpipeDMAMode.NZ2NZ:
        dst = tla.view(Y, (n // 16, 1, 16, 16))
    else:
        dst = tla.view(Y, (rows, cols))
    group: tl.constexpr = 0 if UNIT is not None else None
    with tla.scope.cube():
        a = tla.alloc((k // c0, 1, 16, c0), input_dtype, space="l1", is_mem_unique=True)
        b = tla.alloc((n // c0, k // 16, 16, c0), input_dtype, space="l1", is_mem_unique=True)
        acc = tla.alloc((n // 16, 1, 16, 16), acc_dtype, space="l0c", is_mem_unique=True)
        tla.nd2nz_gm_to_l1(tla.view(A, (16, k)), a)
        tla.nd2nz_gm_to_l1(tla.view(B, (k, n)), b)
        if ACCUMULATE:
            tla.cube.mmad(a,
                          b,
                          acc,
                          clear_accum=True,
                          transpose_b=TRANSPOSE,
                          unit_flag_ctrl=tla.UnitFlagMode.ENABLED_WITHOUT_UPDATE,
                          unit_flag_group_id=0)
            tla.cube.mmad(a,
                          b,
                          acc,
                          clear_accum=False,
                          transpose_b=TRANSPOSE,
                          unit_flag_ctrl=tla.UnitFlagMode.ENABLED_WITH_UPDATE,
                          unit_flag_group_id=0)
        else:
            tla.cube.mmad(a,
                          b,
                          acc,
                          clear_accum=True,
                          transpose_b=TRANSPOSE,
                          unit_flag_ctrl=UNIT,
                          unit_flag_group_id=group)
        tla.fixpipe_l0c_to_dst(acc,
                               dst,
                               dma_mode=DMA,
                               pre_quant=QUANT,
                               pre_relu=RELU,
                               quant_scale=SCALE,
                               unit_flag_ctrl=UNIT,
                               unit_flag_group_id=group,
                               dual_dst_mode=DUAL,
                               sub_block_idx=SUB,
                               channel_split=CHANNEL,
                               c0_pad_en=C0)
    if DEST == "ub":
        with tla.scope.vector():
            tla.store_ub_to_gm(dst, tla.view(Y, (rows, cols)))


def constants(case):
    return dict(OUT=case["out"],
                DMA=getattr(tla.FixpipeDMAMode, case["mode"]),
                RELU=getattr(tla.FixpipePreReluMode, case["relu"]),
                UNIT=case["unit"],
                ACCUMULATE=case["accumulate"],
                DEST=case["dest"],
                QUANT=getattr(tla.FixpipePreQuantMode, case["quant"]),
                SCALE=case["scale"],
                DUAL=getattr(tla.FixpipeDualDstMode, case["dual"]),
                SUB=None if case["sub"] is None else getattr(tla.FixpipeSubBlock, case["sub"]),
                CHANNEL=case["channel"],
                C0=case["c0"],
                TRANSPOSE=case["transpose"])


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    inp = "*i8" if case["out"] == "int8" else "*fp16"
    text = lowered(product, {"A": inp, "B": inp, "Y": "*" + case["out"]}, constants(case), output_dir)
    assert text.count("hivm.hir.nd2nz ") == 2
    assert text.count("hivm.hir.mmadL1 ") == (2 if case["accumulate"] else 1)
    assert text.count("hivm.hir.fixpipe ") == 1
    if case["mode"] != "NZ2NZ":
        assert getattr(tla.FixpipeDMAMode, case["mode"]).value in text
    if case["quant"] != "NO_QUANT":
        assert case["quant"] in text
    if case["relu"] != "NO_RELU":
        assert case["relu"] in text
    if case["unit"] is not None:
        assert f"#hivm.unit_flag<{tla.UnitFlagMode(case['unit']).name}>" in text
    if case["name"] == "gm_f32":
        _check_ir_fixpipe()
        _check_ir_fixpipe_split_and_convert()
        _check_ir_unit_flags()
        _check_invalid_option("unit_flag", "l0c", "gm", "unknown HIVM UnitFlag mode")
        _check_invalid_option("unit_flag_bool", "l0c", "gm", "unit_flag_ctrl must be a compile-time i32 integer")
    print("IR PASS:", OP, case["name"])


def launch(case, compile_only=False):
    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "int8": torch.int8
    }[case["out"]]
    generator = torch.Generator().manual_seed(42)
    k = 32 if case["out"] == "int8" else 16
    n = 32 if case["out"] == "int8" else 16
    a_cpu = (torch.randn(16, k, generator=generator) * 3).to(torch.int8 if k == 32 else torch.float16)
    b_cpu = (torch.randn(k, n, generator=generator) * 3).to(a_cpu.dtype)
    y = torch.full((16, n), -77, dtype=dtype, device="npu")
    a, b = a_cpu.npu(), b_cpu.npu()
    if compile_only:
        return product.warmup(a, b, y, grid=(1, ), enable_tla_mode=get_tla_mode(),
                              **constants(case)), None, None
    kernel = product[(1, )](a, b, y, enable_tla_mode=get_tla_mode(), **constants(case))
    torch.npu.synchronize()
    rhs = b_cpu.float().T if case["transpose"] else b_cpu.float()
    expected = a_cpu.float() @ rhs
    if case["accumulate"]:
        expected *= 2
    if case["scale"] is not None:
        expected *= case["scale"]
    if case["relu"] == "NORMAL_RELU":
        expected = expected.clamp_min(0)
    if case["mode"] == "NZ2DN":
        expected = expected.T.contiguous()
    return kernel, y.cpu(), expected.to(dtype)


@matrix(CONFIGS)
def test_compile(case):
    check_case_binary(launch(case, True)[0])


@matrix(CONFIGS)
def test_run(case):
    if case.get("accuracy_reason"):
        pytest.skip(case["accuracy_reason"])
    _, actual, expected = launch(case)
    rtol, atol = (8e-3, 1e-1) if case["out"] == "bf16" else (1e-3, 3e-2)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=rtol, atol=atol)
    print("ACCURACY PASS:", OP, case["name"])


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
