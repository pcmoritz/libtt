"""Creates and destroys libtt PJRT clients from many threads at once.

Calls the PJRT C API through ctypes, which releases the GIL, so the threads
really run concurrently. Clients share one process-wide instance: a client may
be destroyed on another thread than the one that created it, and concurrent
first clients must initialize the device only once.

As a test it runs next to JAX's client. Run it as a script in a fresh process
to also race the first clients' initialization:

    python client_lifecycle_regression_test.py <path to libtt.so>
"""

import ctypes
import os
import pathlib
import random
import sys
import threading

NUM_THREADS = 8
CHURN_ROUNDS = 20


class ApiVersion(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("major_version", ctypes.c_int),
        ("minor_version", ctypes.c_int),
    ]


class ClientCreateArgs(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("create_options", ctypes.c_void_p),
        ("num_options", ctypes.c_size_t),
        ("kv_get_callback", ctypes.c_void_p),
        ("kv_get_user_arg", ctypes.c_void_p),
        ("kv_put_callback", ctypes.c_void_p),
        ("kv_put_user_arg", ctypes.c_void_p),
        ("client", ctypes.c_void_p),
        ("kv_try_get_callback", ctypes.c_void_p),
        ("kv_try_get_user_arg", ctypes.c_void_p),
    ]


class ClientArgs(ctypes.Structure):
    """PJRT_Client_Destroy_Args, and the input part of other client calls."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("client", ctypes.c_void_p),
    ]


class AddressableDevicesArgs(ctypes.Structure):
    _fields_ = ClientArgs._fields_ + [
        ("addressable_devices", ctypes.c_void_p),
        ("num_addressable_devices", ctypes.c_size_t),
    ]


class ErrorMessageArgs(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("error", ctypes.c_void_p),
        ("message", ctypes.c_char_p),
        ("message_size", ctypes.c_size_t),
    ]


class PluginInitializeArgs(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
    ]


# PJRT_Api: struct_size, extension_start, pjrt_api_version, then the
# functions in pjrt_c_api.h order.
API_FUNCTIONS = [
    "PJRT_Error_Destroy",
    "PJRT_Error_Message",
    "PJRT_Error_GetCode",
    "PJRT_Plugin_Initialize",
    "PJRT_Plugin_Attributes",
    "PJRT_Event_Destroy",
    "PJRT_Event_IsReady",
    "PJRT_Event_Error",
    "PJRT_Event_Await",
    "PJRT_Event_OnReady",
    "PJRT_Client_Create",
    "PJRT_Client_Destroy",
    "PJRT_Client_PlatformName",
    "PJRT_Client_ProcessIndex",
    "PJRT_Client_PlatformVersion",
    "PJRT_Client_Devices",
    "PJRT_Client_AddressableDevices",
]


class Api(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("pjrt_api_version", ApiVersion),
    ] + [(name, ctypes.c_void_p) for name in API_FUNCTIONS]


class Plugin:
    def __init__(self, library_path):
        library = ctypes.CDLL(library_path)
        library.GetPjrtApi.restype = ctypes.POINTER(Api)
        self._api = library.GetPjrtApi().contents
        self._error_message = self._function("PJRT_Error_Message", ErrorMessageArgs)
        self._client_create = self._function("PJRT_Client_Create", ClientCreateArgs)
        self._client_destroy = self._function("PJRT_Client_Destroy", ClientArgs)
        self._addressable_devices = self._function(
            "PJRT_Client_AddressableDevices", AddressableDevicesArgs
        )
        initialize = self._function("PJRT_Plugin_Initialize", PluginInitializeArgs)
        self._check(initialize(ctypes.byref(PluginInitializeArgs(ctypes.sizeof(PluginInitializeArgs)))))

    def _function(self, name, args_type):
        return ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(args_type))(getattr(self._api, name))

    def _check(self, error):
        if error:
            args = ErrorMessageArgs(ctypes.sizeof(ErrorMessageArgs), None, error)
            self._error_message(ctypes.byref(args))
            raise RuntimeError(args.message[: args.message_size].decode())

    def create(self):
        args = ClientCreateArgs(struct_size=ctypes.sizeof(ClientCreateArgs))
        self._check(self._client_create(ctypes.byref(args)))
        return args.client

    def destroy(self, client):
        args = ClientArgs(ctypes.sizeof(ClientArgs), None, client)
        self._check(self._client_destroy(ctypes.byref(args)))

    def num_devices(self, client):
        args = AddressableDevicesArgs(ctypes.sizeof(AddressableDevicesArgs), None, client)
        self._check(self._addressable_devices(ctypes.byref(args)))
        return args.num_addressable_devices


def run_threads(target, items):
    errors = []
    barrier = threading.Barrier(len(items))

    def run(item):
        barrier.wait()
        try:
            target(item)
        except Exception as e:  # pylint: disable=broad-except
            errors.append(e)

    threads = [threading.Thread(target=run, args=(item,)) for item in items]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]


def stress(plugin):
    # Concurrent creates share one client (in a fresh process, the first ones
    # race to initialize the device).
    clients = [None] * NUM_THREADS
    run_threads(lambda i: clients.__setitem__(i, plugin.create()), list(range(NUM_THREADS)))
    assert len(set(clients)) == 1, f"{len(set(clients))} distinct clients"
    assert plugin.num_devices(clients[0]) > 0

    # Destroy each reference on another thread than the one that created it.
    random.Random(0).shuffle(clients)
    run_threads(plugin.destroy, clients)

    # Churn: every thread repeatedly takes and drops a reference.
    def churn(seed):
        rng = random.Random(seed)
        for _ in range(CHURN_ROUNDS):
            client = plugin.create()
            assert plugin.num_devices(client) > 0
            if rng.random() < 0.5:
                threading.Event().wait(rng.random() * 0.01)
            plugin.destroy(client)

    run_threads(churn, list(range(NUM_THREADS)))
    client = plugin.create()
    assert plugin.num_devices(client) > 0
    plugin.destroy(client)


def test_clients_from_many_threads():
    import jax
    import jax_plugins.libtt

    jax.devices()  # JAX's client keeps the device open throughout.
    library = pathlib.Path(jax_plugins.libtt.__file__).with_name("libtt.so")
    stress(Plugin(str(library)))


if __name__ == "__main__":
    stress(Plugin(sys.argv[1]))
    print("OK", flush=True)
    # A worker thread opened the device, and UMD releases its per-chip lock
    # only on the thread that took it, so the device teardown of a normal exit
    # would abort on this thread.
    os._exit(0)
