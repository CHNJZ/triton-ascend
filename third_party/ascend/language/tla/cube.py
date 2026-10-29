# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Cube-core primitives for TLA (hivm.hir.mmadL1).

``mmad`` computes ``C = A x B (+C)`` with A/B in L1 and C in L0C.
Operands may use logical 2-D shapes, or native 4-D NZ shapes
``[column_blocks, row_blocks, row_inner, column_inner]``. The latter keeps
explicit DMA layouts intact through the Affinity template pipeline.
"""
from triton.language.core import builtin, _unwrap_if_constexpr
from .core import buffer
from .dma import _unit_flag_attrs

__all__ = ["mmad"]


@builtin
def mmad(a, b, acc, transpose_b=False, clear_accum=False, unit_flag_ctrl=None, *, unit_flag_group_id=None,
         transpose_a=False, per_channel_bias=None, _semantic=None):
    """L1 MMAD with optional L1 per-channel bias, paired with Fixpipe.

    L0A/L0B and BiasBUF transfers are managed inside the backend template.
    """
    if not (isinstance(a, buffer) and isinstance(b, buffer) and isinstance(acc, buffer)):
        raise TypeError("tla.mmad expects tla.buffer operands for a, b, acc")
    if a.space != "l1" or b.space != "l1":
        raise ValueError("tla.mmad a/b must be L1 buffers (fill via tla.nd2nz_gm_to_l1)")
    if acc.space != "l0c":
        raise ValueError("tla.mmad acc must be an L0C buffer")
    if a.dtype != b.dtype:
        raise ValueError("tla.mmad a and b must have the same element type")
    transpose_a = bool(_unwrap_if_constexpr(transpose_a))
    transpose_b = bool(_unwrap_if_constexpr(transpose_b))
    per_channel_bias = _unwrap_if_constexpr(per_channel_bias)
    if per_channel_bias is not None and not isinstance(per_channel_bias, buffer):
        raise TypeError("tla.mmad per_channel_bias must be a tla.buffer")
    ranks = (len(a.shape), len(b.shape), len(acc.shape))
    if ranks == (2, 2, 2):
        a_rows, a_cols = a.shape
        b_rows, b_cols = b.shape
        acc_m, acc_n = acc.shape
    elif ranks == (4, 4, 4):
        a_rows, a_cols = a.shape[1] * a.shape[2], a.shape[0] * a.shape[3]
        b_rows, b_cols = b.shape[1] * b.shape[2], b.shape[0] * b.shape[3]
        acc_m, acc_n = acc.shape[1] * acc.shape[2], acc.shape[0] * acc.shape[3]
    else:
        raise ValueError("tla.mmad expects all operands to be logical 2-D or physical 4-D NZ buffers")
    m, k = (a_cols, a_rows) if transpose_a else (a_rows, a_cols)
    k2, n = (b_cols, b_rows) if transpose_b else (b_rows, b_cols)
    if k != k2 or (acc_m, acc_n) != (m, n):
        raise ValueError("tla.mmad shape mismatch: a=(M,K), b=(K,N), acc=(M,N)")
    _semantic.builder.create_hivm_mmadl1(a.handle, b.handle, acc.handle, bool(_unwrap_if_constexpr(clear_accum)), m,
                                         k, n, transpose_b,
                                         _unit_flag_attrs(unit_flag_ctrl, unit_flag_group_id, _semantic),
                                         None if per_channel_bias is None else per_channel_bias.handle, transpose_a)
