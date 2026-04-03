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
`/dev/tenstorrent`, then falls back to `/dev/tenstorrent*`, and exposes one
PJRT device per device node it finds. If neither path exists, the client
reports zero devices.

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
