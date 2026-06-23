# msdebug — project overview & status

Single source of truth for the **msdebug** effort on triton-ascend (3.2 baseline,
Huawei Ascend NPU Triton fork). Consolidates the goal, the three layers, the strategy
(and its mid-course correction), the on-hardware gating findings, and the task
roadmap. Branch: `msdebug`.

> Detailed gating mechanics live in `README.md` (this same directory). This file is
> the higher-altitude project view.

---

## 1. Goal

cuda-gdb-style **source-level debugging** for triton-ascend. Two orthogonal dimensions:

1. **Line breakability** — being able to set breakpoints on source lines (colleague's
   work, PRs #151 NOP-injection / #283 disable-opt).
2. **Variable naming** — source variable names (`x`, `offs`, `mask`, …) visible in the
   debugger. In MLIR these are `NameLoc`. **This is the focus of this task.**

The chain we must preserve: source var name → MLIR `NameLoc` in **TTIR** →
through the TTIR→**adapterIR** conversion → `bishengir-compile` → **DWARF** → debugger.

**Hard rule:** maximize alignment with community Triton (3.5/main); deviate only where
NPU-incompatible. `NameLoc` *generation* is always on (not gated). Line/NOP machinery
is gated by `TRITON_DEBUG`.

---

## 2. The three layers

| Layer | What | Status |
|---|---|---|
| **L1** front-end | generate `NameLoc` for source vars into TTIR | **DONE & verified** (commit `313efecad`) |
| **L2** conversion | preserve those locs through TTIR→adapterIR passes (absorption/fold/CSE/region-rebuild) | **PAUSED for naming** (backend can't surface names — §4c); not needed for line-only |
| **L3** end-to-end | wire debug-info into `bishengir-compile`, verify DWARF + single-step | **line dim verified working** (§4d); naming blocked at backend |

**L1 detail (done).** 4 files: `python/src/ir.h`, `python/src/ir.cc`,
`python/triton/compiler/code_generator.py`, `python/triton/language/core.py`. Verified
on vector-add: params + intermediates (`pid/offsets/mask/x/y/output`) carry `NameLoc`
in TTIR.

---

## 3. Where loss happens (empirical: 2338 dumps / 454 kernels)

Losses are highly concentrated — essentially one root cause plus a benign tail:

| Class | Hits | Root cause | Disposition |
|---|---|---|---|
| arange/make_range absorbed | 1154 | index chain folded into `reinterpret_cast` offset | Cat A — `absorbLoc` |
| index/offset absorbed | 564+ | same | Cat A — `absorbLoc` |
| mask compare | 530 | MaskAnalysis folds into access bounds, no surviving op | Cat A — `absorbLoc` |
| program_id / num_programs | 251+ | becomes a function-arg read, no surviving op | orphan — NOP |
| block_ptr/tensor_ptr/descriptor | 40 | OffsetAnalysis absorbs | Cat A — `absorbLoc` |
| SSA alias (x0=xindex / multiple_of) | 76 | one value two names | benign — leave |
| const init / compile-time scalar | 53+ | fold/CSE, or no SSA value | listener / benign |
| elementwise fusion (out=x+y) | ~3 | linalg fusion keeps one loc | listener |
| tl.reshape front-end missing name | ~8 | code_generator gap | front-end fix |
| individual cases | ~10 | CallSiteLoc occupies loc | front-end, low-pri |

**Real "new work" is only 3 things:** fold/CSE listener, reshape front-end naming, and
the individual cases. Everything else is either the *same* `absorbLoc` fix (Cat A) or
benign.

---

## 4. Strategy — and the gating that is reshaping it

### 4a. Planned strategy (from the empirical handoff)
**FusedLoc merge, not NOP/disable-opt.** When an op is absorbed/folded, merge its loc
(incl. `NameLoc`) into the surviving anchor: `survivor.setLoc(FusedLoc{survivor.loc,
lost.loc})` — semantics = `-O` line table. Decisions:
- PR #283 (disable passes) → **retire**, replace with a global fold/CSE Listener.
- PR #151 (NOP) → **downgrade to orphans only** (program_id/num_programs) + gate.
- scf region-rebuild block-arg loc copy (~20 sites) → **suspended** (printer
  blind-spot / implicit block-arg info; verify with `--mlir-print-op-generic`).

This whole plan bets on one assumption: **bishengir penetrates `FusedLoc` to extract
both line and name into DWARF.**

### 4b. GATING RESULT (on hardware — CANN 9.0.0 / Ascend910B4) ⚠️
The bet is (at least partly) **wrong as stated**:

- A real ttadapter carrying `NameLoc`s (+ an injected `FusedLoc`), compiled with
  `--enable-debug-info=true`, **fails the LLVM verifier** exactly on the composite-loc
  ops: `inlinable function call ... must have a !dbg location`.
- Flattening **all** `NameLoc`/`FusedLoc` → plain `FileLineColLoc` makes it **compile**.

**=> The wall is bishengir's own loc→DWARF translation: it does not lower `NameLoc` /
`FusedLoc` into a `!dbg` (unlike upstream MLIR, which recurses). The bottleneck is the
CANN `bishengir`/`hivmc` binary, NOT our conversion layer / `absorbLoc`.**

### 4c. Discriminators — RESOLVED (2026-06-18, Ascend910B4 / CANN 9.0.0)
Ran `run_loc_variants.sh kernel.flat.mlir` (log: `variants_run.log`). From an
already-compiling flattened IR, changed only the one `memref.copy` loc:

| Variant | loc | COMPILE | LINE(4242) | NAME |
|---|---|---|---|---|
| v1b | plain `FileLineColLoc(4242)` | ✅ | **PASS** | n/a |
| v2 | `FusedLoc` of all-`FileLineColLoc` members | ✅ | **FAIL** | FAIL |
| v3 | `NameLoc` as primary loc | ✅ | **PASS** | FAIL |

**Findings:**
- **FusedLoc not penetrated even for line** (v2 LINE FAIL) ⇒ the §4a absorbLoc /
  FusedLoc-merge plan cannot deliver *any* dimension; **retire it as the carrier.**
- **NameLoc-as-primary is safe + line-penetrated** (v3 LINE PASS): bishengir recurses
  into the NameLoc for its inner FileLineColLoc. ⇒ **L1 NameLoc generation is safe to
  keep on; it does not break the line table.**
- **Name never reaches DWARF** (v3 NAME FAIL): zero `DW_TAG_variable`/`formal_parameter`
  in `.debug_info`, even with `LLVM_EXTRACT_DI_LOCAL_VARIABLES=1`. The name string is
  dropped by bishengir's loc→DWARF.

**=> Variable naming is a BACKEND limitation, not a conversion-layer one.** Path forward
for naming: **(A)** vendor fix to `hivmc`/bishengir loc→DWARF, **(B)** lower NameLoc→DWARF
ourselves before bishengir, or **(C)** ship line-only now, names later. The **line**
dimension is unblocked and can proceed independently.

> **Decision (2026-06-18): take path (C) — ship line-only debugging first**; naming
> deferred to a later backend path (A/B). §4d below is the first line-only milestone.

### 4d. Line-only end-to-end — VERIFIED (2026-06-18, real triton compile flow)
No backend change needed: `compiler.py:496-497` & `749-750` already add
`--enable-debug-info=true` whenever `TRITON_DISABLE_LINE_INFO=0`. Compiled the real
`probe_copy_kernel` through the *normal* triton flow (not the manual kit):

```
TRITON_KERNEL_DUMP=1 TRITON_ALWAYS_COMPILE=1 TRITON_DUMP_DIR=./e2e_dump \
  TRITON_DISABLE_LINE_INFO=0 TRITON_DEBUG=1 python3 probe_kernel.py
```

Result on `probe_copy_kernel.npubin`:
- Full DWARF present: `.debug_info / .debug_line / .debug_ranges / .debug_str / .debug_frame`.
- `.debug_line` maps to **real source lines 27 / 29 / 31 / 32** of `probe_kernel.py`
  (the kernel def, `offs`, `tl.load`, `tl.store`) — not all-`1`.
- The ttadapter carries both real line locs *and* L1 NameLocs (`in_ptr/n/offs/val/...`);
  the NameLocs do not break the line table (consistent with v3).

**Gap (feeds line-breakability work):** lines **28** (`pid = program_id`) and **30**
(`mask = offs < n`) were **absent** from the line table — exactly the orphan (program_id)
and absorbed (mask) cases from §3.

### 4e. Line-breakability — loc preservation IMPLEMENTED (2026-06-22)
Master switch: **`LLVM_EXTRACT_DI_LOCAL_VARIABLES=1`** (the msdebug debug-info switch;
off by default, zero impact on production builds). Replaces `TRITON_DEBUG` as the gate —
that flag is too broad (device asserts, prints, …).

- **mask absorbed line (30): DONE & VERIFIED.** The mask sub-expression's loc is now
  threaded onto the generated bound arithmetic in `LoadStoreConverter.cpp` (the real
  producer; twin gated changes in `MemOpConverter.cpp` / `MaskAnalysis.cpp`). With the
  switch on, the probe kernel's `.debug_line` is `27 29 30 31 32`; off, `27 29 31 32`.
  Carrier is plain/NameLoc-wrapped `FileLineColLoc`, which bishengir honors — no FusedLoc.
- **program_id / num_programs orphan: DONE & VERIFIED.** With auto-blockify OFF these
  lower to a bare kernel arg with no surviving op; a value-preserving anchor (`id+0`)
  folds away. Fixed by injecting an **empty side-effecting `llvm.inline_asm` barrier**
  carrying the orphan op's loc in `FunctionConverter.cpp` (gated): it can't be
  folded/DCE'd, changes no value, survives bishengir, and produces a real `.debug_line`
  row. Verified: `probe_compute` line 18 and `probe_kernel` line 28 now present; gate
  off → original behavior; kernels still correct. (Auto-blockify ON already recomputes
  program_id as `div/rem` carrying its loc, so it survives there independently.)
- index/arange absorbed lines: see §coverage audit — dynamic indices already survive;
  constant indices have no executable code (non-breakable, benign). No fix needed.

See `LINE_BREAKABILITY_PLAN.md` for the full design + status and `LINE_DEBUGGING.md` for
usage. Detail: the §3 roadmap's FusedLoc-merge is retired; line preservation uses the
absorbed op's own single loc, which is all bishengir needs for the line table.

---

## 5. Verified facts (correct earlier guesses)

| Thing | Value | Source |
|---|---|---|
| bishengir debug flag | `--enable-debug-info=true` (NOT `-g`/`--mlir-print-debuginfo`) | `backend/compiler.py:496-497` |
| env gate (MUST set) | `TRITON_DISABLE_LINE_INFO=0` (default `"true"` ⇒ debug OFF) | `backend/utils.py:271` |
| bishengir input | `kernel.ttadapter.mlir` | `backend/compiler.py:628-632` |
| ttadapter dump | `TRITON_KERNEL_DUMP=1` | `backend/compiler.py:257` |
| dumped arch on box | `Ascend910B4` | the IR header |

---

## 6. Task roadmap (post-gating, if merge path survives)

Order from the handoff (§4 there). Twin files = both `TritonToLinalg/*` and
`TritonToStructured/*` must change.

- **P0** helper `include/Utils/DebugLocUtils.h`: `absorbLoc` / `absorbLocOf` /
  `absorbChainLocs` (flatten FusedLoc, dedup, cap 8). *Zero-risk, no bishengir dep —
  can write before gating closes.*
- **P0-1** mask (530): `MaskState.absorbedLocs`, push in parseCmp/parseAdd/…, drain at
  Load/StoreConverter anchor. (twin MaskAnalysis.cpp)
- **P0-2** index (arange 1154 + index 564): `BlockData.absorbedLocs`, absorb at
  `reinterpret_cast` in `rewriteLoopOp`. (BlockPtrAnalysis.cpp / PtrAnalysis.cpp)
- **P0-3** global `LocAbsorbListener` in `TritonToLinalgPass` (replaces #283). Real
  reference impl: `lib/DynamicCVPipeline/SplitDataflow/PreserveControlAttrsCanonicalize.cpp:58`
  (`RewriterBase::Listener`, both `Operation*` and `ValueRange` overloads;
  `config.listener=&listener` at :153). Gate on `TRITON_DEBUG`.
- **P1-1** block_ptr (40): `OffsetAnalysis.cpp`. **P1-2** indirect/gather:
  `absorbChainLocs`. **P1-3** program_id orphan NOP + collect #151 (gated).
- **P1-4** reshape front-end (~8). **P2** individual cases + UnknownLoc replacements.

> **Status (2026-06-18): §4c resolved → this roadmap is PAUSED for the *naming*
> dimension.** The absorbLoc/FusedLoc-merge carrier is dead at the backend (FusedLoc not
> penetrated; NameLoc name dropped), so P0/P0-1/P0-2/P0-3 would not surface any name.
> Resume only after a backend path (A vendor fix / B self-lower NameLoc→DWARF) lands.
> The **line** dimension is unblocked and can proceed (NameLoc-as-primary keeps the line
> table; line-only single-step is viable now).

---

## 7. Where things are

- **This kit** (`msdebug_gating/`): `README.md` (gating how-to + FINDINGS),
  `probe_kernel.py`, `inject_fusedloc.py`, `run_gating.sh`, `flatten_locs.py`,
  `run_loc_variants.sh`, `fusedloc_probe.mlir`.
- **Work happens on the NPU box** clone (`/home/j00957045/triton-ascend`) — bishengir,
  readelf, and the triton-adapter build only run there. The Windows dev box has no
  toolchain.
- Analysis tooling (delivered separately, not yet in repo): `check_loc.py` (single
  kernel, CI gate), `check_loc_coverage.py` (batch → analysis.log).
