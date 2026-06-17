#!/usr/bin/env bash
# gating step 2 — compile the FusedLoc-injected ttadapter with bishengir-compile
# (debug info ON) and report, per dimension, whether the FusedLoc penetrated DWARF.
#
# RUN THIS ON THE NPU BOX (needs bishengir-compile + readelf; not available on the
# Windows dev machine where the kit was authored).
#
# Required:
#   ARCH       NPU arch, e.g. Ascend910B2 / Ascend950PR_9579  (matches NPUUtils().get_arch())
#   BISHENGIR  path to bishengir-compile (default: from PATH)
# Usage:
#   ARCH=Ascend910B2 ./run_gating.sh kernel.fusedloc.mlir
set -u

IR="${1:?usage: ARCH=<arch> ./run_gating.sh <fusedloc.mlir>}"
ARCH="${ARCH:-Ascend910B2}"
BISHENGIR="${BISHENGIR:-$(command -v bishengir-compile || true)}"
OUT="${OUT:-kernel.o}"

if [ -z "$BISHENGIR" ]; then
  echo "ERROR: bishengir-compile not found. Set BISHENGIR=/path/to/bishengir-compile"; exit 2
fi

# Per the doc, local-variable DI extraction is gated by this env in the bishengir/llvm path.
export LLVM_EXTRACT_DI_LOCAL_VARIABLES=1

echo "== compiling =="
echo "  $BISHENGIR $IR --target=$ARCH --enable-debug-info=true -o $OUT"
"$BISHENGIR" "$IR" \
  --target="$ARCH" \
  --enable-debug-info=true \
  --enable-hfusion-compile=true \
  --enable-triton-kernel-compile=true \
  -o "$OUT"
rc=$?
# bishengir may emit kernel.o or kernel_reloc.o depending on api version
[ -f "$OUT" ] || OUT="kernel_reloc.o"
if [ $rc -ne 0 ] || [ ! -f "$OUT" ]; then
  echo "ERROR: compile failed (rc=$rc) or object not produced. See stderr above."; exit 3
fi
echo "  -> $OUT"

echo "== inspecting DWARF =="
readelf --debug-dump=decodedline "$OUT" > line.txt 2>/dev/null
readelf --debug-dump=rawline    "$OUT" >> line.txt 2>/dev/null
readelf --debug-dump=info       "$OUT" > info.txt 2>/dev/null

echo
echo "================ GATING VERDICT ================"
# LINE dimension: did the FileLineColLoc member (line 4242) reach the line table?
if grep -qw 4242 line.txt; then
  echo "LINE  : PASS  -> bishengir reads the FileLineColLoc member of FusedLoc"
else
  echo "LINE  : FAIL  -> line 4242 absent from line table (bishengir likely takes only the first/other loc)"
fi
# NAME dimension: did the NameLoc member ('msdbg_probe_var') reach DWARF variable info?
if grep -q "msdbg_probe_var" info.txt; then
  echo "NAME  : PASS  -> bishengir reads the NameLoc member into DWARF (DW_AT_name)"
else
  echo "NAME  : FAIL  -> 'msdbg_probe_var' absent from .debug_info"
  echo "                 (could be: FusedLoc not penetrated, OR NameLoc not lowered to a"
  echo "                  DW_TAG_variable even when penetrated -- inspect info.txt manually)"
fi
echo "==============================================="
echo
echo "Interpretation:"
echo "  LINE+NAME PASS -> FusedLoc-merge strategy is viable; proceed with absorbLoc + listener."
echo "  LINE FAIL      -> backend ignores FusedLoc members; FALL BACK to full NOP route."
echo "  LINE PASS / NAME FAIL -> line table works via fusion, but variable naming needs a"
echo "                 different carrier (e.g. NameLoc as the op's PRIMARY loc, not a fused member)."
echo
echo "raw dumps: line.txt , info.txt   (grep them for context around 4242 / msdbg_probe_var)"
