# Line-breakability implementation (task #4) — making absorbed/orphan lines breakable

Goal: every executable source line of a kernel gets a row in the DWARF `.debug_line`
table, so a debugger can set a breakpoint on it. Gated by `LLVM_EXTRACT_DI_LOCAL_VARIABLES` (off by default,
zero impact on production builds). This is the **line** dimension only — independent of
the (backend-blocked) variable-naming dimension; it relies only on plain `FileLineColLoc`,
which bishengir is verified to honor (gating §4d, v1b/v3).

## Why lines are lost (recap, from the probe kernel)

The TTIR carries every line (27–32). The **TTIR→adapter conversion** drops two kinds:

| Kind | Example | TTIR op | Fate in adapter | Fix |
|---|---|---|---|---|
| **absorbed** | line 30 `mask = offs < n` | `arith.cmpi` (loc=mask:30) | folded into bound arithmetic, which inherits the **memory op's** line (31) | **loc preservation** — tag generated bounds with the absorbed op's own loc |
| **absorbed** | line 29/idx `offs = pid*B + arange` | `make_range`/`addi` | folded into `reinterpret_cast` offset | loc preservation in PtrAnalysis |
| **orphan** | line 28 `pid = program_id` | `tt.get_program_id` | becomes a kernel **arg** read; no surviving op | **NOP injection** — emit a throwaway op carrying the orphan loc |

Key carrier facts:
1. An op has exactly one loc and `FusedLoc` is NOT penetrated by bishengir (v2 FAIL), so
   we cannot "merge" two lines onto one op. Each lost line needs its **own surviving op**
   carrying its plain/NameLoc-wrapped `FileLineColLoc`.
2. **The carrier op must lower to a real machine instruction.** A line only gets a
   `.debug_line` row if some op *on that line* becomes code. Tagging a pure metadata /
   view op (e.g. `memref.reinterpret_cast`, `memref.subview`) does **not** create a
   breakable line — verified: threading the index loc onto the `reinterpret_cast`
   (`BlockPtrAnalysis::rewriteAddPtr`) made the cast carry `offs:19` in the IR but
   produced **no** line-table row, because the cast emits no instruction (and the
   arange offset was a compile-time constant `[0]` — literally no code). That change was
   reverted. mask (A1) works precisely because its carriers (`maxsi/minsi/subi` bound
   arithmetic) are real instructions.

## Part A — absorbed via loc preservation

### A1. mask (≈530 dump hits) — DONE & VERIFIED in this branch
The real producer of the bound arithmetic is
`lib/TritonToLinalg/LoadStoreConverter.cpp` (Load + Store converters): it parsed the
mask with `op.getLoc()` (the load/store line), tagging all generated bound arithmetic
with the *memory* line. Under `LLVM_EXTRACT_DI_LOCAL_VARIABLES` we now parse with
`mask.getLoc()` (the mask expression's own line); `getSubview` keeps the memory `loc` so
the access itself stays on its own line. Twin gated changes also in
`TritonToStructured/MemOpConverter.cpp::createNewMask` and
`TritonToLinalg/MaskAnalysis.cpp::runMaskAnalysisImpl` (alternative mask paths).
Bishengir penetrates the `NameLoc("mask"(…:30))` wrapper to put line 30 in the table.
Verified: line 30 appears in `.debug_line` of the probe kernel (see Status below).

### A2. index / arange (≈1154 + 564 hits) — TODO
`lib/TritonToStructured/PtrAnalysis.cpp` and `lib/TritonToLinalg/BlockPtrAnalysis.cpp`:
when the index chain (`make_range` + `addi` + `muli`) is folded into the
`memref.reinterpret_cast` offset, the cast inherits the memory line. Mirror A1: thread
the index sub-expression's loc (the `make_range` / `addi` value loc) onto the generated
`reinterpret_cast` / offset ops, gated by `LLVM_EXTRACT_DI_LOCAL_VARIABLES`. Same NameLoc-penetration logic.

## Part B — orphans via NOP injection

### B1. program_id / num_programs (≈251 hits) — TODO
`tt.get_program_id` becomes a function argument; nothing survives on its line. Add a
small pass (or fold into an existing late TTIR pass) that, under `LLVM_EXTRACT_DI_LOCAL_VARIABLES`, for each
`tt.get_program_id` / `tt.get_num_programs`, emits a **side-effect-bearing NOP** that
(a) consumes the value so it is not DCE'd, (b) carries the op's `FileLineColLoc`, and
(c) lowers to a real machine instruction so it gets a line-table address. Candidate
carriers to validate on hardware (pick the first that survives bishengir):
- an `llvm`-level `donothing`/volatile marker injected late, or
- a `hivm`/adapter no-op annotation op if one exists, or
- a trivial scalar arith op whose result is threaded into a kept value.
Validate exactly like A1: confirm line 28 appears in `.debug_line`.

### B2. other UnknownLoc / individual cases — TODO (low priority)
Per OVERVIEW §3 tail: const-init/compile-time scalars (benign), reshape front-end gap
(~8, a front-end fix), CallSiteLoc occupying loc (~10). Address after A/B land.

## Gating & cache
- All behavior is behind `mlir::triton::tools::getBoolEnv("LLVM_EXTRACT_DI_LOCAL_VARIABLES")`.
- Consider adding `LLVM_EXTRACT_DI_LOCAL_VARIABLES` to `CACHE_INVALIDATING_ENV_VARS`
  (`include/triton/Tools/Sys/GetEnv.hpp`) so toggling it busts the compile cache; the
  Python `debug` kwarg already differentiates the kernel hash, so this is belt-and-suspenders.

## Verification loop (per increment)
```bash
cd msdebug_gating
rm -rf e2e_dump ~/.triton/cache
TRITON_KERNEL_DUMP=1 TRITON_ALWAYS_COMPILE=1 TRITON_DUMP_DIR=./e2e_dump \
  TRITON_DISABLE_LINE_INFO=0 LLVM_EXTRACT_DI_LOCAL_VARIABLES=1 python3 probe_kernel.py
BIN=$(find ./e2e_dump -name '*.npubin' | head -1)
readelf --debug-dump=decodedline "$BIN" | awk '$2 ~ /^[0-9]+$/{print $2}' | sort -un
# expect lines 27 28 29 30 31 32 once A+B complete (currently 27 29 31 32)
```

## Coverage audit (2026-06-22) — what is still fixable at the adapter level?
Ran richer kernels to find any *executable* op that loses its line (the only fixable
class). With the mask fix applied:

| Kernel | Lines present | Lines lost | Lost-line nature |
|---|---|---|---|
| `probe_compute.py` (load, ×4 compute, store, mask) | def, offs, mask, load, all compute, **store** | `program_id` only | orphan (backend) |
| `probe_loop.py` (scf.for reduction) | def, **for**, mask, load, acc+=, store | `arange`, `zeros` init | compile-time constants (no code) |

**Conclusion (updated 2026-06-22): line coverage is now COMPLETE for every line that has
executable code.** With the mask fix (A1) + the orphan inline-asm anchor (B1):
- `probe_compute`: `17 18 19 20 21 22 23 24 25 26` — every source line.
- `probe_kernel`: `27 28 29 30 31 32` — every source line.
- `probe_loop`: `8 11 12 13 14 15` — every line except the two below.

The only lines that can ever be missing are **compile-time constants**
(`tl.arange(0,B)` folded to a constant offset, `tl.zeros` init): there is *no machine
instruction* for them (the `zeros` line isn't even in the TTIR), so they are
fundamentally non-breakable — not a bug, and an inline-asm anchor there would be
meaningless (nothing executes on that line). Everything that lowers to an instruction —
compute, load, store, mask, index, loop/branch, **program_id** — is now breakable.

## Status (verified 2026-06-22)
- **A1 mask loc-preservation: DONE & VERIFIED.** Implemented in
  `lib/TritonToLinalg/LoadStoreConverter.cpp` (the real producer of the bound
  arithmetic — `mstate.parse(mask, maskLoc, …)` in both Load and Store converters),
  with twin gated changes in `TritonToStructured/MemOpConverter.cpp::createNewMask` and
  `TritonToLinalg/MaskAnalysis.cpp::runMaskAnalysisImpl`. `LLVM_EXTRACT_DI_LOCAL_VARIABLES` added to
  `CACHE_INVALIDATING_ENV_VARS` (else `getBoolEnv` asserts). Probe kernel `.debug_line`:
  `LLVM_EXTRACT_DI_LOCAL_VARIABLES=1` → `27 29 30 31 32` (mask line 30 now breakable);
  `LLVM_EXTRACT_DI_LOCAL_VARIABLES=0` → `27 29 31 32` (unchanged). Kernel correctness holds both ways.
  > Note: the first attempt edited `TritonToLinalg/MaskAnalysis.cpp` only and had no
  > effect — for this kernel the bounds are produced by `LoadStoreConverter`, not by
  > `runMaskAnalysisImpl`. The fix had to go where the surviving ops are actually built.

- **B1 program_id / num_programs orphan: DONE & VERIFIED.** When auto-blockify is OFF
  (`autoBlockifySize == 1`), `tt.get_program_id` lowers to a bare kernel argument
  (`GetProgramIDConverter`, `FunctionConverter.cpp`) with no surviving op. A
  value-preserving anchor (`id + 0`) is an identity and gets folded away (confirmed
  earlier). **Fix: inject an empty *side-effecting* `llvm.inline_asm` barrier** carrying
  the orphan op's loc, gated, in both `GetProgramIDConverter` and `GetNumProgramsConverter`
  (`anchorOrphanLine`). It (a) can't be folded/DCE'd (`has_side_effects=true`),
  (b) changes no value (no operands/results, empty asm string — correctness-safe),
  (c) survives the whole pipeline + bishengir (the LLVM dialect is already a legal/loaded
  dialect of `TritonToLinalgPass`, and `LLVM::InlineAsmOp` is used elsewhere in the
  backend), and (d) produces a real `.debug_line` row. Verified: `probe_compute` line 18
  and `probe_kernel` line 28 now appear; gate off → original; all probe kernels still
  pass their `allclose` asserts. (Earlier "backend wall" conclusion was wrong — an empty
  side-effecting inline-asm is the fold-resistant, code-emitting NOP, and it works from
  the adapter.) Auto-blockify ON also makes program_id survive independently via
  `div/rem` (`AutoBlockify.cpp:262-270`).

- A2 index/arange loc-preservation: NOT YET — and **simple kernels do not reproduce the
  loss**. Empirically (2026-06-22, `probe_arange.py`, pure `offs = tl.arange`):
  the index line (21) and mask line (22) both survive, but the **load (23) and store (24)
  lines are lost** — a *third* absorption pattern: in a pure copy the load/store fuse
  into a single `memref.copy` / `materialize_in_destination`, which inherit the
  offs/mask locs (`#loc9 = val(21)`, `#loc5 = 22`) instead of the load/store op's own
  line. So the real absorption taxonomy on this stack is at least:
    1. **mask → bound arithmetic** (FIXED, A1),
    2. **index/arange → reinterpret_cast offset** (the §3 1154+564 cases; needs a
       multi-dim / strided / block-pointer kernel to reproduce — not a trivial 1-D copy),
    3. **load/store → fused memref.copy** (pure elementwise/copy kernels lose the
       load/store line; fix = tag the copy with the load loc and the
       materialize/store with the store loc, in `LoadStoreConverter`/`MemOpConverter`).
  Each pattern needs its own reproducer + producer-site fix (same loc-threading
  technique as A1). Recommend driving these with the **real failing kernels** from the
  original 2338-dump / 454-kernel analysis rather than synthetic probes.

  **Refined understanding of the index/arange class (2338-dump §3 "1154+564"):**
  split by whether the index has executable code:
  - **dynamic index** (offset = real arithmetic, e.g. `pid*BLOCK+arange`): the line is
    already carried by the offset arithmetic ops (`muli`/`index_cast`) — verified on
    `probe_kernel.py` (`offs` line 29 present with the switch off *and* on). No fix
    needed for the line dimension; loss in §3 was about the *name*, not the line.
  - **constant index** (offset folds to a compile-time constant, e.g. pure
    `tl.arange(0,B)`): there is **no executable instruction** for the index, so the line
    is fundamentally non-breakable (like an orphan) — tagging the view op does nothing
    (see carrier fact #2). This is a benign case, not a fixable one.
  So "extend A1 to index/arange" only has teeth where a *dynamic* index op exists but
  was mis-tagged with the memory line; that needs a real kernel exhibiting it.
- B2 tail: not started.
