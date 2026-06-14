load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES")
load("@rules_cc//cc:cc_toolchain_config_lib.bzl", "feature", "flag_group", "flag_set", "tool_path")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/toolchains:cc_toolchain_config_info.bzl", "CcToolchainConfigInfo")

def _sfpi_cc_toolchain_config_impl(ctx):
    compile_actions = [
        ACTION_NAMES.c_compile,
        ACTION_NAMES.cpp_compile,
        ACTION_NAMES.assemble,
        ACTION_NAMES.preprocess_assemble,
    ]

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        toolchain_identifier = "sfpi-riscv-tt-elf",
        host_system_name = "local",
        target_system_name = "riscv-tt-elf",
        target_cpu = "riscv-tt",
        target_libc = "unknown",
        compiler = "gcc",
        abi_version = "unknown",
        abi_libc_version = "unknown",
        tool_paths = [
            tool_path(name = "ar", path = "compiler/bin/riscv-tt-elf-ar"),
            tool_path(name = "cpp", path = "compiler/bin/riscv-tt-elf-cpp"),
            tool_path(name = "gcc", path = "compiler/bin/riscv-tt-elf-g++"),
            tool_path(name = "gcov", path = "compiler/bin/riscv-tt-elf-gcov"),
            tool_path(name = "ld", path = "compiler/bin/riscv-tt-elf-g++"),
            tool_path(name = "nm", path = "compiler/bin/riscv-tt-elf-nm"),
            tool_path(name = "objcopy", path = "compiler/bin/riscv-tt-elf-objcopy"),
            tool_path(name = "objdump", path = "compiler/bin/riscv-tt-elf-objdump"),
            tool_path(name = "strip", path = "compiler/bin/riscv-tt-elf-strip"),
        ],
        cxx_builtin_include_directories = [
            "external/+http_archive+sfpi/compiler/riscv-tt-elf/include/c++/15.1.0",
            "external/+http_archive+sfpi/compiler/riscv-tt-elf/include/c++/15.1.0/riscv-tt-elf",
            "external/+http_archive+sfpi/compiler/riscv-tt-elf/include/c++/15.1.0/backward",
            "external/+http_archive+sfpi/compiler/lib/gcc/riscv-tt-elf/15.1.0/include",
            "external/+http_archive+sfpi/compiler/lib/gcc/riscv-tt-elf/15.1.0/include-fixed",
            "external/+http_archive+sfpi/compiler/riscv-tt-elf/include",
        ],
        features = [
            feature(
                name = "no_canonical_prefixes",
                enabled = True,
                flag_sets = [
                    flag_set(
                        actions = compile_actions,
                        flag_groups = [flag_group(flags = ["-no-canonical-prefixes"])],
                    ),
                ],
            ),
        ],
    )

sfpi_cc_toolchain_config = rule(
    implementation = _sfpi_cc_toolchain_config_impl,
    provides = [CcToolchainConfigInfo],
)
