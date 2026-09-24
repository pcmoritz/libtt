"""Exercise upstream physical discovery using JAX coordination, without MPI."""

import argparse
import ctypes
import os
import traceback

import jax
from jax._src import distributed

parser = argparse.ArgumentParser()
parser.add_argument("--coordinator", required=True)
parser.add_argument("--rank", type=int, required=True)
parser.add_argument("--size", type=int, default=2)
parser.add_argument("--library", required=True)
parser.add_argument("--output", required=True)
parser.add_argument(
    "--test-config", help="Run the upstream mesh-socket payload test after discovery."
)
parser.add_argument(
    "--mesh-descriptor", help="TT-Metal mesh graph for the payload test."
)
parser.add_argument(
    "--kernel-root",
    help="TT-Metal source root containing upstream socket test kernels.",
)
args = parser.parse_args()
if args.test_config:
    if not args.mesh_descriptor or not args.kernel_root:
        parser.error("--test-config requires --mesh-descriptor and --kernel-root")
    os.environ.update(
        LIBTT_FABRIC_TEST_CONFIG=args.test_config,
        TT_MESH_GRAPH_DESC_PATH=args.mesh_descriptor,
        TT_METAL_KERNEL_PATH=args.kernel_root,
        TT_MESH_ID=str(args.rank),
        TT_MESH_HOST_RANK="0",
    )
jax.distributed.initialize(
    coordinator_address=args.coordinator,
    num_processes=args.size,
    process_id=args.rank,
    initialization_timeout=60,
)
client = distributed.global_state.client
buffers = []
errors = []


@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t)
def put(key, data, length):
    try:
        client.key_value_set_bytes(key.decode(), ctypes.string_at(data, length))
        return 0
    except Exception as error:  # noqa: BLE001 - exceptions cannot cross a ctypes callback
        errors.append(error)
        traceback.print_exc()
        return 1


@ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t))
def get(key, length):
    try:
        value = client.blocking_key_value_get_bytes(key.decode(), 60000)
        buffer = ctypes.create_string_buffer(value)
        buffers.append(buffer)
        length[0] = len(value)
        return ctypes.addressof(buffer)
    except Exception as error:  # noqa: BLE001 - exceptions cannot cross a ctypes callback
        errors.append(error)
        traceback.print_exc()
        return None


library = ctypes.CDLL(args.library)
library.libtt_discover.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    type(put),
    type(get),
    ctypes.c_char_p,
]
library.libtt_discover.restype = ctypes.c_int
try:
    result = library.libtt_discover(
        args.rank, args.size, put, get, args.output.encode()
    )
    if result or errors:
        raise RuntimeError(f"Fabric diagnostic failed on rank {args.rank}")
    print(
        f"PASS rank {args.rank}: discovery and requested payload tests completed without MPI",
        flush=True,
    )
finally:
    jax.distributed.shutdown()
