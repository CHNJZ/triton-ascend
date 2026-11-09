# A5 Proton / msprof op experiment

This experiment preserves `proton.record` in A5 TTAdapter IR. NPU-IR lowers
the records to `proton_scope_boundary` / DFX region instructions. A3 keeps
the existing scratch-buffer instrumentation path.

The experiment covers the SIMD/TTAdapter compilation path, not SIMT-only
or TLA compilation. It does not automatically launch msprof or convert its
output into a Proton Chrome Trace.

## Requirements

- Build TA with `TRITON_BUILD_PROTON=ON`.
- Use an NPU-IR compiler supporting A5 Proton lowering and
  `--proton-metadata-output` (validated with PR #4704, `115945de0b94`).
- Source the CANN environment, and use msprof op with A5 PipeTimeline support.
- Do not preload MSPTI for this collection path.

## Run

From the repository root, with the built TA package on `PYTHONPATH`:

```bash
unset LD_PRELOAD
msprof op --aic-metrics=PipeTimeline \
  --kernel-name=proton_a5_probe --launch-count=1 --warm-up=0 \
  --output="$PWD/msprof-a5" \
  --application="python3 $PWD/third_party/ascend/unittest/proton_ut/proton_a5_msprof_op.py --n 256 --output $PWD/proton-a5-ir"
```

The Python program enables instrumentation using `proton.start()` and checks
the numerical result, eight retained `proton.record` operations, and zero
profiling scratch size. `metadata.json` stores NPU-IR's region ID/name mapping
in `proton_metadata`. Use `--n 4097` to also exercise a masked tail.

The Python program's `status=passed` means kernel execution and IR checks
passed, **not that msprof collected scope timing data**. Check msprof's logs
and output separately. `proton-host.chrome_trace` is expected to be empty
because this A5 path does not return records through Proton's buffer parser.

## Validation on 2026-11-09

- Built against main-dev `b51a67e7e` using CANN 9.1.0 and the existing PR #4704 compiler.
- Direct A5 execution passed for N=256 and N=4097. LLVM IR contains the eight
  `proton_scope_boundary` calls with the exported start/end region IDs.
- Related Host tests: 127 passed, 28 skipped. A3 routing is covered by Host
  tests; no new A3 hardware run was performed.
- Actual msprof op collection was blocked before the application started:
  the installed tool queries device 0 for the platform, and this machine's
  device 0 query returned error 87. Devices 1-6 returned valid A5 information.
  `ASCEND_RT_VISIBLE_DEVICES=1` did not change this tool-level query.
- No scope performance data was collected. The tool returned exit code 0
  despite printing errors, so exit status alone must not be used as success.
