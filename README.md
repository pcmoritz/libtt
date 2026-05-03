# libtt

Minimal Rust `cdylib` that builds `libtt.so` and exports the official PJRT C
plugin entrypoint:

```c
const PJRT_Api* GetPjrtApi(void);
```

## Supported operations

- Client create/destroy
- Device and addressable-device listing
- Device description, topology, and minimal memory queries needed for JAX startup

Device discovery is intentionally minimal right now: the library scans
`/dev/tenstorrent` and exposes one PJRT device per device node it finds.
If the path does not exist, the client reports zero devices.

The discovery path now feeds an internal device abstraction modeled on
`blackhole-py`'s `device.py`, including board selection (`p100`/`p150`),
worker-core layout, command-queue core coordinates, and harvested DRAM-bank
metadata. On Linux, probing now follows the older `blackhole-py` driver path:
it opens `/dev/tenstorrent/<n>` and uses the Tenstorrent driver ioctls to read
ARC telemetry through a temporary TLB mapping.

## DRAM Helpers

The crate now also exposes Linux-only DRAM helpers modeled on
`blackhole-py`'s `dram.py`:

- `device::Device::open(local_hardware_id)` opens `/dev/tenstorrent/<n>` and
  exposes board metadata together with DRAM allocation helpers.
- `dram::Allocator` performs DRAM allocation and raw tiled page reads/writes.
- `device::Device::alloc_write(...)` accepts an untiled tensor payload, tilizes
  it into Blackhole tile order, writes it to DRAM, and returns a `DramBuffer`.

Example:

```rust
use libtt::device::Device;
use libtt::dram::DType;

let mut device = Device::open(0)?;
let rows = 32usize;
let cols = 64usize;
let data = vec![0u16; rows * cols]
    .into_iter()
    .flat_map(|value| value.to_le_bytes())
    .collect::<Vec<_>>();
let buffer = device.alloc_write(&data, DType::Float16, &[rows, cols], "weights")?;
let roundtrip = device.dram_read(&buffer)?;
assert_eq!(roundtrip, data);
```

## Build

```bash
bazel build //:tt
```

This produces a single shared library containing the Rust PJRT implementation
and, for the Bazel `//:tt` target, the C++ MLIR frontend as well.

On Linux the output is `libtt.so`; on macOS it is `libtt.dylib`.

The default Rust unit tests run with:

```bash
bazel test //:tt_test
```

## Optional MLIR Frontend

The PJRT layer can analyze StableHLO/MLIR programs through a C++ frontend while
keeping the Rust PJRT interface and TT runtime unchanged.

The Bazel build now pulls the pinned StableHLO and LLVM/MLIR sources into the
Bazel dependency graph directly. It does not depend on the full `xla` Bazel
module.

The easiest path is:

```bash
bazel build //:tt
```

The current lowering is still intentionally small, but StableHLO parsing,
bytecode deserialization, and MLIR pass plumbing live on the C++ side rather
than as Rust string matching.

The first Bazel build is expected to download and analyze a large upstream
dependency graph because LLVM/MLIR and StableHLO are now built through Bazel
instead of a preinstalled local prefix.

## Regenerating PJRT Bindings

The checked-in Rust bindings live in `src/pjrt_bindings.rs` and are generated
from the vendored OpenXLA header at:

```text
third_party/openxla/xla/pjrt/c/pjrt_c_api.h
```

To regenerate them after updating that header, run:

```bash
cargo run --manifest-path xtask/Cargo.toml -- update-pjrt-bindings
```

Notes:

- This uses the standalone `xtask` helper crate, so the library build does not
  depend on Cargo or `bindgen`.
- The regeneration helper has its own lockfile at `xtask/Cargo.lock`.
- You will need a working `libclang`/`clang` installation for `bindgen`.
- After regenerating, review the diff in `src/pjrt_bindings.rs` and run
  `bazel test //:tt_test //:tt_mlir_test`.

## Using It

Load the shared library, resolve `GetPjrtApi`, and use the official
`pjrt_c_api.h` definitions from OpenXLA.

```c
#include "xla/pjrt/c/pjrt_c_api.h"

const PJRT_Api* api = GetPjrtApi();
```
