"""A5 DFX-region probe. Run under msprof op; its output is the collection result."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import triton.profiler as proton
import triton.profiler.language as pl

pl.enable_semantic("triton")


@triton.jit
def proton_a5_probe(x, y, out, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with pl.scope("kernel"):
        with pl.scope("load"):
            a = tl.load(x + offsets, offsets < n, other=0)
            b = tl.load(y + offsets, offsets < n, other=0)
        with pl.scope("compute"):
            result = a + b
        with pl.scope("store"):
            tl.store(out + offsets, result, offsets < n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.n <= 0 or args.iterations <= 0:
        parser.error("n and iterations must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(args.device)
    x = torch.arange(args.n, device="npu", dtype=torch.float32)
    y = torch.ones_like(x)
    out = torch.empty_like(x)
    session = proton.start(str(args.output / "proton-host"), backend="instrumentation", data="trace")
    try:
        for _ in range(args.iterations):
            compiled = proton_a5_probe[(triton.cdiv(args.n, 256), )](x, y, out, args.n, BLOCK=256, debug=True)
            torch.npu.synchronize()
        metadata = compiled.metadata._asdict()
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        for name, value in compiled.asm.items():
            if isinstance(value, str):
                (args.output / f"kernel.{name}").write_text(value, encoding="utf-8")
        assert metadata["proton_collection"] == "msprof_op"
        assert metadata.get("profile_scratch_size", 0) == 0
        assert compiled.asm["ttadapter"].count("proton.record") == 8
        assert "__builtin_proton_circular_store" not in compiled.asm["ttadapter"]
        assert "proton_metadata" in metadata
    finally:
        proton.finalize(session, output_format="chrome_trace")
    torch.testing.assert_close(out, x + y)
    print(
        json.dumps({
            "status": "passed", "n": args.n, "iterations": args.iterations, "collector": "msprof op", "scope_metadata":
            metadata["proton_metadata"]
        }), flush=True)


if __name__ == "__main__":
    main()
