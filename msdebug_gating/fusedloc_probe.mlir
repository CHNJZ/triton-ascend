// gating — SELF-CONTAINED CANDIDATE (secondary path).
//
// Hand-authored minimal ttadapter (post triton-to-linalg) for a vector-copy kernel,
// with a sentinel FusedLoc already injected on the surviving store (memref.copy).
//
// PREFER the probe_kernel.py + inject_fusedloc.py path: this hand-authored IR is a
// best-effort guess at the exact entry-adapter shape bishengir-compile accepts, and
// has NOT been validated against a real bishengir build (authored on a Windows box
// without the toolchain). If bishengir rejects it (arg types / required attributes /
// missing alloc-and-copy lowering), regenerate from a real dump instead of fighting it.
//
// The FusedLoc under test (two members):
//   loc(fused["msdbg_probe.py":4242:7, "msdbg_probe_var"])
//     - FileLineColLoc "...":4242:7  -> LINE dimension
//     - NameLoc        "msdbg_probe_var" -> NAME dimension
//
// Adjust hacc.target to your arch before compiling.

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  func.func @probe_copy_kernel(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: i32)
      attributes {global_kernel = "local", mix_mode = "mix"} {
    %src = memref.reinterpret_cast %arg0 to offset: [0], sizes: [256], strides: [1]
        : memref<*xf32> to memref<256xf32, strided<[1]>>
    %dst = memref.reinterpret_cast %arg1 to offset: [0], sizes: [256], strides: [1]
        : memref<*xf32> to memref<256xf32, strided<[1]>>
    %alloc = memref.alloc() : memref<256xf32>
    memref.copy %src, %alloc : memref<256xf32, strided<[1]>> to memref<256xf32>
    memref.copy %alloc, %dst : memref<256xf32> to memref<256xf32, strided<[1]>> loc(fused["msdbg_probe.py":4242:7, "msdbg_probe_var"])
    return
  }
}
