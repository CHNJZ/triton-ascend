#!/usr/bin/env python3
"""
gating step 1 — inject a sentinel FusedLoc onto a SURVIVING anchor op in a real
ttadapter dump, so we can later check whether bishengir-compile propagates both
members of the FusedLoc into DWARF.

The FusedLoc we inject has exactly two members:
  - a FileLineColLoc with a sentinel line  -> tests the LINE dimension
  - a NameLoc with a sentinel variable name -> tests the NAME dimension

  loc(fused["msdbg_probe.py":4242:7, "msdbg_probe_var"])

We OVERWRITE the anchor op's existing loc so the sentinels (4242 / msdbg_probe_var)
are the ONLY occurrences in the whole module -> any later grep hit is unambiguous.

Usage:
  python3 inject_fusedloc.py kernel.ttadapter.mlir [-o kernel.fusedloc.ttadapter.mlir] [--anchor memref.copy]

Why an injection probe (instead of waiting for the real conversion to produce a
FusedLoc): Layer-2 absorbLoc is NOT written yet — the whole point of gating is to
prove the bishengir backend CONSUMES FusedLoc before we invest in PRODUCING it.
"""
import argparse
import re
import sys

SENTINEL_LINE = 4242
SENTINEL_COL = 7
SENTINEL_FILE = "msdbg_probe.py"
SENTINEL_NAME = "msdbg_probe_var"
FUSED = f'loc(fused["{SENTINEL_FILE}":{SENTINEL_LINE}:{SENTINEL_COL}, "{SENTINEL_NAME}"])'

# Anchor candidates in priority order: ops that survive lowering and carry a real
# access. memref.copy is the canonical store anchor after triton-to-linalg.
DEFAULT_ANCHORS = ["memref.copy", "memref.store", "linalg.", "bufferization.materialize"]

# strip a trailing ` loc(...)` on a single-line op (loc is always the last token)
TRAILING_LOC = re.compile(r"\s+loc\(.*\)\s*$")


def pick_anchor_line(lines, anchors):
    for pat in anchors:
        for i, ln in enumerate(lines):
            # skip comments / CHECK lines
            stripped = ln.lstrip()
            if stripped.startswith("//") or stripped.startswith("#"):
                continue
            if pat in ln:
                return i, pat
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--output")
    ap.add_argument("--anchor", action="append", default=None,
                    help="op substring to anchor on (repeatable); overrides defaults")
    args = ap.parse_args()

    anchors = args.anchor if args.anchor else DEFAULT_ANCHORS
    text = open(args.input, "r", encoding="utf-8").read()
    lines = text.splitlines()

    idx, used = pick_anchor_line(lines, anchors)
    if idx is None:
        sys.exit(f"ERROR: no anchor op found (tried {anchors}). "
                 f"Inspect the IR and pass --anchor <substring> for a surviving op.")

    original = lines[idx]
    base = TRAILING_LOC.sub("", original.rstrip())
    lines[idx] = f"{base} {FUSED}"

    out = args.output or (args.input.rsplit(".mlir", 1)[0] + ".fusedloc.mlir")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[inject] anchor op: '{used}' at line {idx+1}")
    print(f"[inject]   before: {original.strip()}")
    print(f"[inject]   after : {lines[idx].strip()}")
    print(f"[inject] sentinels -> LINE={SENTINEL_LINE}  NAME={SENTINEL_NAME}")
    print(f"[inject] wrote: {out}")


if __name__ == "__main__":
    main()
