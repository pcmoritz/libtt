load("@rules_cc//cc:defs.bzl", "cc_library")

def _sfpi_platform_transition_impl(settings, attr):
    return {
        "//command_line_option:platforms": attr.platform,
    }

_sfpi_platform_transition = transition(
    implementation = _sfpi_platform_transition_impl,
    inputs = [],
    outputs = ["//command_line_option:platforms"],
)

def _sfpi_cc_object_files_impl(ctx):
    dep = ctx.attr.dep
    if type(dep) == "list":
        dep = dep[0]
    object_files = dep[OutputGroupInfo].compilation_outputs
    return [
        DefaultInfo(files = object_files),
        OutputGroupInfo(compilation_outputs = object_files),
    ]

_sfpi_cc_object_files = rule(
    implementation = _sfpi_cc_object_files_impl,
    attrs = {
        "dep": attr.label(cfg = _sfpi_platform_transition),
        "platform": attr.string(default = "@sfpi//:blackhole_firmware_platform"),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
)

def sfpi_cc_objects(name, visibility = None, platform = "@sfpi//:blackhole_firmware_platform", **kwargs):
    cc_library(
        name = name + "_cc",
        visibility = ["//visibility:private"],
        **kwargs
    )

    _sfpi_cc_object_files(
        name = name,
        dep = ":" + name + "_cc",
        platform = platform,
        visibility = visibility,
    )
