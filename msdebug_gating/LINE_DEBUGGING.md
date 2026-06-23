# Line-level source debugging for triton-ascend (msdebug)

Status as of **2026-06-18**. This is the **line-only** delivery of the msdebug effort:
source **line** mapping works end-to-end; source **variable names** do **not** (blocked
in the CANN `bishengir` backend — see `OVERVIEW.md` §4c). Use this to set breakpoints on
source lines and single-step a kernel; do not expect `x`/`offs`/`mask` to appear by name.

## What works / what doesn't

| Capability | State | Why |
|---|---|---|
| `.debug_line` → source `.py` lines | ✅ works | bishengir lowers plain `FileLineColLoc` (incl. the one inside a `NameLoc`) to DWARF line table |
| breakpoint on a line backed by a surviving op | ✅ works | the op carries the source `FileLineColLoc` |
| breakpoint on an *absorbed* line (`mask`) | ✅ works (switch on) | mask sub-expr loc threaded onto the surviving bound arithmetic (`LoadStoreConverter`) |
| breakpoint on an *orphan* line (`program_id`/`num_programs`) | ✅ works (switch on) | empty side-effecting `llvm.inline_asm` anchor injected with the orphan loc (`FunctionConverter`) |
| breakpoint on a *compile-time constant* line (`tl.arange(0,B)`, `tl.zeros` init) | ❌ impossible | no machine instruction is emitted for the line — nothing to break on (not a bug) |
| variable names in the debugger | ❌ blocked | bishengir drops `NameLoc` names; emits no `DW_TAG_variable` (gating §4c). Needs a backend path (A/B) |

## Required environment

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export TRITON_DISABLE_LINE_INFO=0   # REQUIRED. Default is "true" => debug info OFF.
                                    # Setting 0 makes compiler.py pass --enable-debug-info=true
export LLVM_EXTRACT_DI_LOCAL_VARIABLES=1   # msdebug master switch: enables the line-breakability
                                           # loc-preservation (mask absorbed-line, etc.)
export TRITON_ALWAYS_COMPILE=1      # bypass cache so your edits actually recompile
```

`TRITON_DISABLE_LINE_INFO=0` is the load-bearing one: `third_party/ascend/backend/utils.py:271`
defaults it to `"true"` (debug OFF), and `compiler.py:496-497` / `749-750` only add
`--enable-debug-info=true` when it is not disabled.

## Compile a kernel with debug info

Just run the kernel with the env above. To also dump the IR + the compiled object:

```bash
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=./dump \
TRITON_DISABLE_LINE_INFO=0 TRITON_ALWAYS_COMPILE=1 LLVM_EXTRACT_DI_LOCAL_VARIABLES=1 \
  python3 your_kernel.py
```

The compiled object is `<dump>/<hash>/<kernel_name>.npubin` (an ELF with DWARF).

## Inspect the DWARF (sanity check before debugging)

```bash
BIN=$(find ./dump -name '*.npubin' | head -1)

# line table -> should show your real source line numbers, not all "1"
readelf --debug-dump=decodedline "$BIN" | less

# which DWARF sections exist
readelf -S "$BIN" | grep debug
```

Reference run on `probe_copy_kernel` (this kit's `probe_kernel.py`) with the switch on
produces line-table entries for **every** source line: **27 28 29 30 31 32** — including
`program_id` (28, via the inline-asm orphan anchor) and `mask` (30, via loc preservation).
With the switch off it is the unenhanced `27 29 31 32`.

## Known gaps / next work

1. **Compile-time-constant lines are not breakable** (e.g. a line that is only
   `tl.arange(0,B)` folded to a constant offset, or `tl.zeros` init). These emit no
   machine instruction at all, so there is nothing to break on — fundamental, not a bug.
   Every line that lowers to an instruction (compute, load, store, mask, index,
   loop/branch, program_id) is breakable with the switch on.
2. **No variable names.** Backend limitation; tracked separately (paths A/B in
   `OVERVIEW.md` §4c). The line dimension does not depend on it.

## See also
- `OVERVIEW.md` — project goal, layers, gating results, roadmap.
- `README.md` — the FusedLoc/NameLoc penetration gating and how it was run.
- `variants_run.log` — raw v1b/v2/v3 discriminator output.
