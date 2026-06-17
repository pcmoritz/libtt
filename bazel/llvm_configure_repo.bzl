# Derived from llvm-project's utils/bazel/configure.bzl to materialize
# @llvm-project directly from this repo's pinned llvm archive.

LLVM_TARGETS = [
    "AArch64",
    "NVPTX",
    "X86",
]

MAX_TRAVERSAL_STEPS = 1000000

MLIR_TARGET_CPP_BUILD = """

cc_library(
    name = "MLIRTargetCpp",
    srcs = [
        "lib/Target/Cpp/TranslateRegistration.cpp",
        "lib/Target/Cpp/TranslateToCpp.cpp",
    ],
    hdrs = ["include/mlir/Target/Cpp/CppEmitter.h"],
    includes = ["include"],
    deps = [
        ":ControlFlowDialect",
        ":EmitCDialect",
        ":FuncDialect",
        ":IR",
        ":Support",
        ":TranslateLib",
        "//llvm:Support",
    ],
)
"""

def _overlay_directories(repository_ctx):
    src_root = repository_ctx.path(Label("@llvm-raw//:WORKSPACE")).dirname
    overlay_root = src_root.get_child("utils/bazel/llvm-project-overlay")
    target_root = repository_ctx.path(".")

    stack = ["."]
    for _ in range(MAX_TRAVERSAL_STEPS):
        rel_dir = stack.pop()
        overlay_dirs = {}

        for entry in overlay_root.get_child(rel_dir).readdir():
            name = entry.basename
            full_rel_path = rel_dir + "/" + name
            if entry.is_dir:
                stack.append(full_rel_path)
                overlay_dirs[name] = None
            else:
                repository_ctx.symlink(
                    overlay_root.get_child(full_rel_path),
                    target_root.get_child(full_rel_path),
                )

        for src_entry in src_root.get_child(rel_dir).readdir():
            if src_entry.basename in overlay_dirs:
                continue
            repository_ctx.symlink(
                src_entry,
                target_root.get_child(rel_dir + "/" + src_entry.basename),
            )

        if not stack:
            return

    fail("overlay_directories exceeded MAX_TRAVERSAL_STEPS ({})".format(MAX_TRAVERSAL_STEPS))

def _extract_cmake_settings(repository_ctx, llvm_cmake):
    values = {
        "CMAKE_CXX_STANDARD": None,
        "LLVM_VERSION_MAJOR": None,
        "LLVM_VERSION_MINOR": None,
        "LLVM_VERSION_PATCH": None,
        "LLVM_VERSION_SUFFIX": None,
    }

    llvm_cmake_path = repository_ctx.path(llvm_cmake)
    for line in repository_ctx.read(llvm_cmake_path).splitlines():
        setfoo = line.partition("(")
        if setfoo[1] != "(" or setfoo[0].strip().lower() != "set":
            continue

        kv = setfoo[2].strip()
        sep = kv.find(" ")
        if sep < 0:
            continue

        key = kv[:sep]
        if key == "LLVM_REQUIRED_CXX_STANDARD":
            key = "CMAKE_CXX_STANDARD"
            values[key] = None
        if key not in values or values[key] != None:
            continue

        values[key] = kv[sep:].strip().partition(")")[0].partition(" ")[0]

    return values

def _vars_bzl_content(values):
    content = "# Generated from pinned llvm-project sources.\n\n"
    for key, value in values.items():
        content += '{} = "{}"\n'.format(key, value)

    content += "\nllvm_vars = {\n"
    for key, value in values.items():
        content += '    "{}": "{}",\n'.format(key, value)
    content += "}\n"
    return content

def _targets_bzl(name, values):
    return "{} = {}".format(name, values)


def _llvm_configure_impl(repository_ctx):
    _overlay_directories(repository_ctx)
    repository_ctx.file(
        "mlir/BUILD.bazel",
        content = repository_ctx.read("mlir/BUILD.bazel") + MLIR_TARGET_CPP_BUILD,
    )

    vars = _extract_cmake_settings(repository_ctx, "llvm/CMakeLists.txt")
    version = _extract_cmake_settings(repository_ctx, "cmake/Modules/LLVMVersion.cmake")
    vars.update({key: value for key, value in version.items() if value != None})
    vars["LLVM_VERSION"] = "{}.{}.{}".format(
        vars["LLVM_VERSION_MAJOR"],
        vars["LLVM_VERSION_MINOR"],
        vars["LLVM_VERSION_PATCH"],
    )
    vars["PACKAGE_VERSION"] = "{}.{}.{}{}".format(
        vars["LLVM_VERSION_MAJOR"],
        vars["LLVM_VERSION_MINOR"],
        vars["LLVM_VERSION_PATCH"],
        vars["LLVM_VERSION_SUFFIX"],
    )

    repository_ctx.file("vars.bzl", content = _vars_bzl_content(vars))

    repository_ctx.file("llvm/targets.bzl", content = _targets_bzl("llvm_targets", LLVM_TARGETS))

    bolt_targets = [target for target in LLVM_TARGETS if target in ["AArch64", "RISCV", "X86"]]
    repository_ctx.file("bolt/targets.bzl", content = _targets_bzl("bolt_targets", bolt_targets))

llvm_configure = repository_rule(
    implementation = _llvm_configure_impl,
    local = True,
    configure = True,
)
