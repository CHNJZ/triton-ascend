#!/usr/bin/env python3
"""
Rewrite every NameLoc / FusedLoc in an MLIR file to a plain FileLineColLoc.

Used by the gating to isolate the failure cause: a real ttadapter carrying
NameLocs (+ an injected FusedLoc) makes bishengir-compile --enable-debug-info=true
fail the LLVM verifier, but the SAME IR with all composite locs flattened COMPILES.
That pins the failure on bishengir's loc->DWARF translation not understanding
NameLoc/FusedLoc, rather than on anything in our conversion layer.

Usage:
    python3 flatten_locs.py <in.mlir> <out.mlir>

Verify it cleaned everything:
    grep -c 'fused\\|"(' <out.mlir>   # expect 0
"""
import re
import sys

PLAIN = 'loc("flatten.py":1:1)'


def flatten(text: str) -> str:
    # FusedLoc:  loc(fused[...])  or  loc(fused<metadata>[...])
    text = re.sub(r'loc\(fused(<[^>]*>)?\[[^\]]*\]\)', PLAIN, text)
    # NameLoc with a child:  loc("name"(<inner>))  -- inner has no nested parens here
    text = re.sub(r'loc\("[^"]*"\([^()]*\)\)', PLAIN, text)
    return text


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python3 flatten_locs.py <in.mlir> <out.mlir>")
    src, dst = sys.argv[1], sys.argv[2]
    out = flatten(open(src, "r", encoding="utf-8").read())
    open(dst, "w", encoding="utf-8").write(out)
    print(f"[flatten] wrote {dst}")


if __name__ == "__main__":
    main()
