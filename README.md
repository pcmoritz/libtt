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
metadata. Right now the crate only populates the portion it can discover
directly from the device-node layout.

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

To dump the PJRT device and memory debug strings directly from the plugin:

```bash
python examples/pjrt_debug_dump.py target/release/libtt.so
```

To trace device discovery and Linux probing, set `LIBTT_LOG=1` before loading the
plugin. Logs are written to stderr.
