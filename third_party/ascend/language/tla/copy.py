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
"""Buffer-to-buffer DMA. No tensor materialization or bind_buffer annotation."""
from triton.language.core import builtin
from .core import buffer
from . import dma


@builtin
def copy(src, dst, _semantic=None):
    if not isinstance(src, buffer) or not isinstance(dst, buffer):
        raise TypeError("tla.copy expects two tla.buffer operands; pointer/tensor arguments are unsupported")
    if (src.space, dst.space) == ("gm", "ub"):
        dma.load_gm_to_ub(src, dst, _semantic=_semantic)
    elif (src.space, dst.space) == ("ub", "gm"):
        dma.store_ub_to_gm(src, dst, _semantic=_semantic)
    elif (src.space, dst.space) == ("gm", "l1"):
        dma.nd2nz_gm_to_l1(src, dst, _semantic=_semantic)
    elif src.space == "l0c":
        mode = dma.FixpipeDMAMode.NZ2ND if len(dst.shape) == 2 else dma.FixpipeDMAMode.NZ2NZ
        dma.fixpipe_l0c_to_dst(src, dst, dma_mode=mode, _semantic=_semantic)
    elif (src.space, dst.space) == ("ub", "l1"):
        dma.copy_ub_to_l1(src, dst, _semantic=_semantic)
    elif (src.space, dst.space) == ("l1", "ub"):
        dma.copy_l1_to_ub(src, dst, _semantic=_semantic)
    elif (src.space, dst.space) == ("l1", "bt"):
        dma.copy_l1_to_bt(src, dst, _semantic=_semantic)
    else:
        raise ValueError(f"tla.copy unsupported space pair {(src.space, dst.space)}")
