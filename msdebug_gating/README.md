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

## Files
- `probe_kernel.py` — tiny kernel to dump a real ttadapter (step 0)
- `inject_fusedloc.py` — inject sentinel FusedLoc onto a surviving op (step 1)
- `run_gating.sh` — compile + readelf + verdict (step 2)
- `fusedloc_probe.mlir` — self-contained hand-authored candidate (fallback)
