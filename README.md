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

## Sysmem

The crate also exposes Linux-only pinned host memory modeled on
`blackhole-py`'s `hw.py::Sysmem`:

```rust
use libtt::sysmem::Sysmem;

let mut sysmem = Sysmem::with_size(0, 1 << 20)?;
sysmem.write(0, &[1, 2, 3, 4])?;
assert_eq!(&sysmem.as_slice()[..4], &[1, 2, 3, 4]);
println!("sysmem noc_addr=0x{:x}", sysmem.noc_addr());
```

## Build

```bash
cargo build --release
```

On Linux the shared library will be written to `target/release/libtt.so`. On
macOS the corresponding artifact is `target/release/libtt.dylib`.

## Using It

Load the shared library, resolve `GetPjrtApi`, and use the official
`pjrt_c_api.h` definitions from OpenXLA.

```c
#include "xla/pjrt/c/pjrt_c_api.h"

const PJRT_Api* api = GetPjrtApi();
```
