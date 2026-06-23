# msdebug gating — does bishengir-compile penetrate `FusedLoc`?

**Why this exists.** The whole Layer-2 strategy (retire PR #283, downgrade #151 to
orphans, use `absorbLoc` to merge a lost op's loc into a survivor via
`FusedLoc{survivor.loc, lost.loc}`) bets on one unverified assumption:

> bishengir-compile reads **both** members of a `FusedLoc` into DWARF — the
> `FileLineColLoc` for the line table **and** the `NameLoc` for the variable name —
> the way it recurses into `CallSiteLoc`.

If it only takes the first/primary loc, merging is useless and we fall back to the
full-NOP route. **Do this before writing any `absorbLoc` code.**

> ⚠️ This kit was authored on a Windows dev box with **no toolchain**. It cannot be
> run there. Run it on the **NPU server** (needs `bishengir-compile` + `readelf`).

---

## FINDINGS SO FAR (2026-06-17, CANN 9.0.0 / Ascend910B4) — read this first

**The wall is bishengir itself, NOT our conversion layer.**

Running a real vector-copy ttadapter (carrying Layer-1 NameLocs + an injected
sentinel FusedLoc) through `bishengir-compile --enable-debug-info=true`:

| Input | Result |
|---|---|
| real ttadapter with NameLoc + FusedLoc | **compile FAILS** |
| failing ops | exactly the composite-loc ops: the `linalg.fill` with a `NameLoc`, and the `memref.copy` with the injected `FusedLoc` |
| error | `inlinable function call in a function with debug info must have a !dbg location` (LLVM verifier) |
| every plain-`FileLineColLoc` op | fine |
| flatten ALL NameLoc/FusedLoc → plain `FileLineColLoc` (`flatten_locs.py`) | **compile PASSES**, emits `kernel.o` |

**Conclusion.** bishengir's own loc→DWARF translation only understands plain
`FileLineColLoc`. It does **not** lower `NameLoc` or `FusedLoc` into a `!dbg`
location (unlike upstream MLIR, which recurses into them) — so ops carrying those
locs reach the LLVM verifier with no `!dbg` and the build aborts *before* DWARF is
emitted. The bottleneck is the CANN `bishengir`/`hivmc` binary, **not** the
absorbLoc/listener work in the adapter.

### DISCRIMINATOR RESULTS (2026-06-18, Ascend910B4 / CANN 9.0.0) — RESOLVED
Ran `run_loc_variants.sh kernel.flat.mlir` (full log: `variants_run.log`). From an
already-compiling `kernel.flat.mlir`, changed only the one `memref.copy` loc:

| Variant | loc on the copy | COMPILE | LINE (4242) | NAME (msdbg_probe_var) |
|---|---|---|---|---|
| v1b control | plain `loc("probe.py":4242:7)` | ✅ | **PASS** | n/a |
| v2 | `loc(fused[<FileLineColLoc>, <FileLineColLoc>])` | ✅ | **FAIL** | FAIL |
| v3 | `loc("msdbg_probe_var"("probeN.py":4242:7))` | ✅ | **PASS** | FAIL |

Three hard conclusions:

1. **FusedLoc is a dead carrier.** v2 — a FusedLoc whose members are *all*
   FileLineColLocs — does **not** put line 4242 in the line table. bishengir does not
   penetrate FusedLoc even for the *line* dimension. ⇒ the planned absorbLoc /
   FusedLoc-merge strategy (retire #283, merge a lost op's loc into a survivor via
   `FusedLoc{...}`) **cannot deliver even the line dimension** on this bishengir.
2. **NameLoc-as-primary is safe and LINE-penetrated.** v3 compiles, and bishengir
   *recurses into* the NameLoc to pull its inner FileLineColLoc(4242) into the line
   table (LINE PASS). ⇒ keeping L1 NameLoc generation on does **not** break the line
   table — the front-end loc work is safe to ship.
3. **The name never reaches DWARF.** v3 NAME FAIL: `.debug_info` has **zero**
   `DW_TAG_variable` / `DW_TAG_formal_parameter`, and `msdbg_probe_var` is absent —
   even with `LLVM_EXTRACT_DI_LOCAL_VARIABLES=1` set. bishengir's loc→DWARF in this
   CANN version simply does not lower a NameLoc's name string into a local-variable DIE.

**Verdict: variable naming is a BACKEND problem, not a conversion-layer one.** No amount
of absorbLoc/listener work in the adapter can surface names, because (a) the FusedLoc
carrier dies before DWARF and (b) even a surviving NameLoc's name is dropped by
bishengir. The naming dimension now reduces to README options:
- **(A)** vendor fix to `hivmc`/bishengir loc→DWARF (emit DW_TAG_variable from NameLoc).
- **(B)** lower NameLoc → DWARF (or a DI-bearing form bishengir *does* read) ourselves,
  *before* handing IR to bishengir.
- **(C)** names infeasible on this bishengir version — ship line-only debugging now.

The **line** dimension is unaffected and works (plain FileLineColLoc, incl. the one
inside a NameLoc). Line-only single-step debugging can proceed independently.

### Corrections to the handoff doc (confirmed on hardware)
- debug flag is `--enable-debug-info=true` (not `-g` / `--mlir-print-debuginfo`).
- it is OFF by default; must set `TRITON_DISABLE_LINE_INFO=0`.
- the dumped arch on this box is **Ascend910B4** (use it for `ARCH=`).

---

## Verified facts (from the repo, not guesses)

| Thing | Value | Source |
|---|---|---|
| bishengir debug-info flag | `--enable-debug-info=true` | `backend/compiler.py:496-497` |
| Env gate (MUST set) | `TRITON_DISABLE_LINE_INFO=0` (default `"true"` ⇒ debug info **OFF**) | `backend/utils.py:271` |
| bishengir input | the `kernel.ttadapter.mlir` file | `backend/compiler.py:628-632` |
| ttadapter dump | `TRITON_KERNEL_DUMP=1` (via dump_manager) | `backend/compiler.py:257` |
| local-var DI extraction | `LLVM_EXTRACT_DI_LOCAL_VARIABLES=1` | handoff doc Layer 3 |

The handoff doc guessed `-g` / `--mlir-print-debuginfo` — **wrong**; the real flag is
`--enable-debug-info=true`.

## Sentinels

Injected `FusedLoc`: `loc(fused["msdbg_probe.py":4242:7, "msdbg_probe_var"])`
- line **4242** (won't occur naturally) ⇒ unambiguous LINE grep
- name **msdbg_probe_var** ⇒ unambiguous NAME grep

## How to run (recommended path)

On the NPU box:

```bash
# 0. produce a guaranteed-valid ttadapter from a real kernel
TRITON_KERNEL_DUMP=1 TRITON_ALWAYS_COMPILE=1 TRITON_DUMP_DIR=./dump \
  TRITON_DISABLE_LINE_INFO=0 python3 probe_kernel.py
ADAPTER=$(find ./dump -name kernel.ttadapter.mlir | head -1)

# 1. inject the sentinel FusedLoc onto a surviving anchor op (memref.copy)
python3 inject_fusedloc.py "$ADAPTER" -o kernel.fusedloc.mlir

# 2. compile with debug info + inspect DWARF, print PASS/FAIL per dimension
ARCH=Ascend910B2 BISHENGIR=$(command -v bishengir-compile) \
  ./run_gating.sh kernel.fusedloc.mlir
```

Fallback if you don't want to run a kernel: use the hand-authored
`fusedloc_probe.mlir` directly in step 2 (best-effort; regenerate from a real dump if
bishengir rejects it).

## Reading the verdict

- **LINE PASS + NAME PASS** → FusedLoc-merge is viable. Proceed: `DebugLocUtils.h` +
  P0-1 mask + P0-2 index + P0-3 listener.
- **LINE FAIL** → backend ignores fused members. Abandon merge; fall back to the full
  NOP route (old plan in `msdebug-nameloc-task.md`).
- **LINE PASS / NAME FAIL** → line table works through fusion, but variable naming
  needs a different carrier. Re-run with the NameLoc as the op's **primary** loc
  (`loc("msdbg_probe_var"(...))`, not a fused member) to localize whether the loss is
  fusion-penetration or NameLoc→DW_TAG_variable lowering. Inspect `info.txt`.

## Variant tests (the pending v1b / v2 / v3 discriminators)

`flatten_locs.py` rewrites every NameLoc/FusedLoc in an IR to a plain FileLineColLoc
(this is what made the real dump compile). `run_loc_variants.sh` then builds v1b/v2/v3
from a flattened IR and runs each through `run_gating.sh`:

```bash
# from a real dump already injected + flattened:
python3 flatten_locs.py kernel.fusedloc.mlir kernel.flat.mlir
ARCH=Ascend910B4 bash run_loc_variants.sh kernel.flat.mlir
```

For each variant it prints whether it **compiled** and the LINE/NAME verdict. Read the
outcome map in "FINDINGS SO FAR" above.

## Files
- `probe_kernel.py` — tiny kernel to dump a real ttadapter (step 0)
- `inject_fusedloc.py` — inject sentinel FusedLoc onto a surviving op (step 1)
- `run_gating.sh` — compile + readelf + verdict (step 2)
- `flatten_locs.py` — rewrite all NameLoc/FusedLoc → plain FileLineColLoc (makes the
  real dump compile; isolates composite-loc as the failure cause)
- `run_loc_variants.sh` — build + run the v1b/v2/v3 discriminator variants
- `fusedloc_probe.mlir` — self-contained hand-authored candidate (fallback)
