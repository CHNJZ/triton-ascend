#!/usr/bin/env bash
# Build and run the v1b / v2 / v3 loc discriminator variants from a flattened IR,
# to learn HOW FAR bishengir's loc->DWARF support goes (see README "FINDINGS SO FAR").
#
# Precondition: <flat.mlir> already compiles (all composite locs flattened) and
# contains exactly one `memref.copy` line we can rewrite. Produce it with:
#     python3 flatten_locs.py kernel.fusedloc.mlir kernel.flat.mlir
#
# Usage:
#     ARCH=Ascend910B4 bash run_loc_variants.sh kernel.flat.mlir
set -u

FLAT="${1:?usage: ARCH=<arch> bash run_loc_variants.sh <flat.mlir>}"
ARCH="${ARCH:-Ascend910B4}"
HERE="$(cd "$(dirname "$0")" && pwd)"

if ! grep -q 'memref\.copy' "$FLAT"; then
  echo "ERROR: no memref.copy line in $FLAT to rewrite"; exit 2
fi

run_variant() {
  local name="$1" newloc="$2" src="$3"
  echo
  echo "########################################################"
  echo "# VARIANT $name  ->  loc = $newloc"
  echo "########################################################"
  # replace the trailing loc(...) on the memref.copy line with the variant loc
  sed "/memref\.copy/ s| loc([^)]*)\?.*| $newloc|" "$FLAT" > "$src"
  ARCH="$ARCH" bash "$HERE/run_gating.sh" "$src"
}

# v1b: plain FileLineColLoc 4242 -> expect COMPILE + LINE PASS (method sanity)
run_variant "v1b-plain"    'loc("probe.py":4242:7)'                       kernel.v1b.mlir
# v2 : FusedLoc whose members are all FileLineColLocs -> does LINE survive a fused?
run_variant "v2-fused"     'loc(fused["probeA.py":4242:7, "probeB.py":99:3])' kernel.v2.mlir
# v3 : NameLoc as the primary loc -> does bishengir accept NameLoc at all?
run_variant "v3-nameloc"   'loc("msdbg_probe_var"("probeN.py":4242:7))'    kernel.v3.mlir

echo
echo "Done. For each variant note: (1) did it COMPILE  (2) LINE/NAME verdict."
echo "See README 'FINDINGS SO FAR' for the outcome map."
