# JAX 0.7.1's pinned XLA revision.
_XLA_COMMIT = "31eb6029e8973337ef6c99810c27e1d807791c11"
_HEADERS = {
    "xla/backends/profiler/plugin/profiler_c_api.h": "ad15b638c6d7820b52ae1aa8ff71f64f692c118ef7e7debe6a9f4aa9e7727732",
    "xla/pjrt/c/pjrt_c_api_profiler_extension.h": "9ae899234c03c1a117ef88481f8a242bdfa6adf0014c88c1485aee3ffd26c70b",
}

def _xla_profiler_headers_repository_impl(repository_ctx):
    for path, sha256 in _HEADERS.items():
        repository_ctx.download(
            output = path,
            sha256 = sha256,
            url = "https://raw.githubusercontent.com/openxla/xla/{}/{}".format(
                _XLA_COMMIT,
                path,
            ),
        )

    repository_ctx.file("BUILD.bazel", """
load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "pjrt_profiler_c_api",
    hdrs = glob(["xla/**/*.h"]),
    visibility = ["//visibility:public"],
)
""")

xla_profiler_headers_repository = repository_rule(
    implementation = _xla_profiler_headers_repository_impl,
)
