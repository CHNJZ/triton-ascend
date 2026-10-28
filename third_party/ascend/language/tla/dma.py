"""Explicit DMA routes and options, passed through to HIVM buffer operations."""
from enum import Enum, IntEnum

from triton.language.core import builtin, _unwrap_if_constexpr
from .core import buffer


class PadMode(Enum):
    PadNull = "PadNull"
    PadFirstElem = "PadFirstElem"
    PadValue = "PadValue"


class EvictionPolicy(Enum):
    EvictNormal = "EvictNormal"
    EvictFirst = "EvictFirst"
    EvictLast = "EvictLast"


class AtomicKind(Enum):
    NONE = "none"
    ADD = "add"
    MAX = "max"
    MIN = "min"
    AND = "and"
    OR = "or"
    XOR = "xor"
    UMAX = "umax"
    UMIN = "umin"


class UnitFlagMode(IntEnum):
    DISABLED = 0
    RESERVED = 1
    ENABLED_WITHOUT_UPDATE = 2
    ENABLED_WITH_UPDATE = 3


class FixpipeDMAMode(Enum):
    NZ2ND = "nz2nd"
    NZ2DN = "nz2dn"
    NZ2NZ = "normal"


class FixpipeDualDstMode(Enum):
    NO_DUAL = "NO_DUAL"
    ROW_SPLIT = "ROW_SPLIT"
    COLUMN_SPLIT = "COLUMN_SPLIT"


class FixpipeSubBlock(Enum):
    SUB_BLOCK_0 = "sub_block_0"
    SUB_BLOCK_1 = "sub_block_1"


class FixpipePreQuantMode(Enum):
    NO_QUANT = "NO_QUANT"
    F322F16 = "F322F16"
    S322I8 = "S322I8"
    QF322F32_PRE = "QF322F32_PRE"
    F322BF16 = "F322BF16"


class FixpipePreReluMode(Enum):
    NO_RELU = "NO_RELU"
    NORMAL_RELU = "NORMAL_RELU"
    LEAKY_RELU = "LEAKY_RELU"
    P_RELU = "P_RELU"


def _buffers(src, dst):
    if not isinstance(src, buffer) or not isinstance(dst, buffer):
        raise TypeError("TLA DMA expects two tla.buffer operands")


def _enum_attr(value, enum, mnemonic, semantic):
    value = _unwrap_if_constexpr(value)
    if value is None:
        return None
    if not isinstance(value, enum):
        raise TypeError(f"expected {enum.__name__}, got {value!r}")
    return semantic.builder.parse_attr(f"#hivm.{mnemonic}<{value.value}>")


def _integer(value, name, bits=64):
    value = _unwrap_if_constexpr(value)
    if type(value) is not int or not -(2**(bits - 1)) <= value < 2**(bits - 1):
        raise ValueError(f"{name} must be a compile-time i{bits} integer")
    return value


def _boolean(value, name):
    value = _unwrap_if_constexpr(value)
    if type(value) is not bool:
        raise TypeError(f"{name} must be a compile-time bool")
    return value


def _scalar(value, semantic, dtype=None):
    value = _unwrap_if_constexpr(value)
    if value is None:
        return None
    result = semantic.to_tensor(value)
    if result.shape:
        raise TypeError("DMA scalar operands must be scalar values")
    if dtype is not None:
        result = semantic.cast(result, dtype)
    elif not result.dtype.is_int():
        raise TypeError("DMA padding counts must be integers")
    return result.handle


def _emit(kind, src, dst, semantic, attrs=None, operands=()):
    attrs = {name: value for name, value in (attrs or {}).items() if value is not None}
    semantic.builder.create_tla_dma(kind, src.handle, dst.handle, attrs, list(operands))


def _unit_flag_attrs(control, group_id, semantic):
    """Encode a static native mode; conditional modes remain backend-managed."""
    control = _unwrap_if_constexpr(control)
    attrs = {}
    if control is not None:
        if isinstance(control, UnitFlagMode):
            control = control.value
        control = _integer(control, "unit_flag_ctrl", bits=32)
        mode = semantic.builder.get_tla_unit_flag_attr(control)
        attrs["unit_flag_mode"] = semantic.builder.get_array_attr([mode])
    if _unwrap_if_constexpr(group_id) is not None:
        group_id = _integer(group_id, "unit_flag_group_id")
        attrs["unit_flag_group_id"] = semantic.builder.parse_attr(f"{group_id} : i64")
    return attrs


@builtin
def copy_ub_to_l1(src, dst, _semantic=None):
    _buffers(src, dst)
    _emit("copy", src, dst, _semantic)


@builtin
def copy_l1_to_ub(src, dst, _semantic=None):
    """L1 NZ to UB ND, using HIVM's l12ub operation."""
    _buffers(src, dst)
    _emit("l12ub", src, dst, _semantic)


@builtin
def copy_l1_to_bt(src, dst, _semantic=None):
    """Copy to BiasBUF; the backend n operand is the last source dimension."""
    _buffers(src, dst)
    if not src.shape:
        raise ValueError("copy_l1_to_bt requires a source extent to derive n")
    _emit("l12bt", src, dst, _semantic, {"n": _semantic.builder.parse_attr(f"{src.shape[-1]} : i64")})


@builtin
def copy_l1_to_fb(src, dst, _semantic=None):
    _buffers(src, dst)
    raise NotImplementedError("AscendNPU-IR at PR #4403 has no L1->FixBUF DMA operation; "
                              "hivm.hir.copy rejects this address-space pair")


@builtin
def load_gm_to_l1(src, dst, _semantic=None):
    """Plain GM to L1 load, without ND-to-NZ conversion."""
    _buffers(src, dst)
    _emit("load", src, dst, _semantic, operands=(None, None, None))


@builtin
def nd2nz_gm_to_l1(src, dst, l2_cache_ctrl=None, _semantic=None):
    _buffers(src, dst)
    cache = _unwrap_if_constexpr(l2_cache_ctrl)
    attrs = {}
    if cache is not None:
        cache = _integer(cache, "l2_cache_ctrl", bits=32)
        attrs["l2_cache_mode"] = _semantic.builder.get_int32_attr(cache)
    _emit("nd2nz", src, dst, _semantic, attrs)


@builtin
def load_gm_to_ub(src, dst, pad_mode=None, pad_value=None, left_padding_num=None, right_padding_num=None,
                  eviction_policy=None, _semantic=None):
    _buffers(src, dst)
    attrs = {
        "pad_mode": _enum_attr(pad_mode, PadMode, "padmode", _semantic),
        "eviction_policy": _enum_attr(eviction_policy, EvictionPolicy, "eviction_policy", _semantic),
    }
    _emit("load", src, dst, _semantic, attrs, (_scalar(
        pad_value, _semantic, src.dtype), _scalar(left_padding_num, _semantic), _scalar(right_padding_num, _semantic)))


@builtin
def store_ub_to_gm(src, dst, atomic_kind=None, _semantic=None):
    _buffers(src, dst)
    attrs = {"atomic_kind": _enum_attr(atomic_kind, AtomicKind, "atomic_kind", _semantic)}
    _emit("store", src, dst, _semantic, attrs)


@builtin
def fixpipe_l0c_to_dst(src, dst, dma_mode=FixpipeDMAMode.NZ2NZ, dual_dst_mode=FixpipeDualDstMode.NO_DUAL,
                       sub_block_idx=FixpipeSubBlock.SUB_BLOCK_0, pre_quant=FixpipePreQuantMode.NO_QUANT,
                       pre_relu=FixpipePreReluMode.NO_RELU, channel_split=False, c0_pad_en=True, quant_scale=None,
                       unit_flag_ctrl=None, *, unit_flag_group_id=None, _semantic=None):
    _buffers(src, dst)
    attrs = {
        "dma_mode": _enum_attr(dma_mode, FixpipeDMAMode, "dma_mode", _semantic),
        "dual_dst_mode": _enum_attr(dual_dst_mode, FixpipeDualDstMode, "fixpipe_dual_dst_mode", _semantic),
        "sub_block_idx": _enum_attr(sub_block_idx, FixpipeSubBlock, "fixpipe_sub_block", _semantic),
        "pre_quant": _enum_attr(pre_quant, FixpipePreQuantMode, "fixpipe_pre_quant_mode", _semantic),
        "pre_relu": _enum_attr(pre_relu, FixpipePreReluMode, "fixpipe_pre_relu_mode", _semantic),
        "channel_split": _semantic.builder.get_bool_attr(_boolean(channel_split, "channel_split")),
        "c0_pad_en": _semantic.builder.get_bool_attr(_boolean(c0_pad_en, "c0_pad_en")),
    }
    attrs.update(_unit_flag_attrs(unit_flag_ctrl, unit_flag_group_id, _semantic))
    from triton.language import float32
    _emit("fixpipe", src, dst, _semantic, attrs, (_scalar(quant_scale, _semantic, float32), ))


@builtin
def copy_l1_to_l0(src, dst, is_transpose=False, actual_batch=None, *, k_part_idx=None, k_part=None, k_part_ceil=None,
                  k_part_loop=None, k_part_actual=None, m=None, _semantic=None):
    """Pass native L12L0 K-partition fields; actual_batch has no backend equivalent."""
    _buffers(src, dst)
    if _unwrap_if_constexpr(actual_batch) is not None:
        raise NotImplementedError("actual_batch has no field in HIVM L12L0Op; "
                                  "specify the native k_part_* and m attributes")
    fields = dict(k_part_idx=k_part_idx, k_part=k_part, k_part_ceil=k_part_ceil, k_part_loop=k_part_loop,
                  k_part_actual=k_part_actual, m=m)
    attrs = {}
    for name, value in fields.items():
        if _unwrap_if_constexpr(value) is None:
            raise ValueError(f"copy_l1_to_l0 requires native backend field {name}")
        number = _integer(value, name)
        attrs[name] = _semantic.builder.parse_attr(f"{number} : i64")
    attrs["is_transposed"] = _semantic.builder.get_bool_attr(_boolean(is_transpose, "is_transpose"))
    _emit("l12l0", src, dst, _semantic, attrs)


@builtin
def set_ctrl(enable, idx, _semantic=None):
    """Write the specified hardware CTRL bit through hivm.hir.set_ctrl."""
    enable = _boolean(enable, "enable")
    idx = _integer(idx, "idx")
    _semantic.builder.create_tla_set_ctrl(enable, idx)


@builtin
def _read_ctrl(_semantic=None):
    """Internal readback used to verify set_ctrl without assuming initial bits."""
    from triton.language import tensor, uint64
    return tensor(_semantic.builder.create_tla_read_ctrl(), uint64)
