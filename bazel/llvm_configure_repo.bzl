# Derived from llvm-project's utils/bazel/configure.bzl to let this repo
# materialize @llvm-project directly from the pinned llvm archive.

DEFAULT_TARGETS = [
    "AArch64",
    "X86",
]

MAX_TRAVERSAL_STEPS = 1000000

def _overlay_directories(repository_ctx):
    src_root = repository_ctx.path(
        Label("@{}//:WORKSPACE".format(repository_ctx.attr.llvm_raw_repo)),
    ).dirname
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

    values["LLVM_VERSION"] = "{}.{}.{}".format(
        values["LLVM_VERSION_MAJOR"],
        values["LLVM_VERSION_MINOR"],
        values["LLVM_VERSION_PATCH"],
    )
    values["PACKAGE_VERSION"] = "{}.{}.{}{}".format(
        values["LLVM_VERSION_MAJOR"],
        values["LLVM_VERSION_MINOR"],
        values["LLVM_VERSION_PATCH"],
        values["LLVM_VERSION_SUFFIX"],
    )
    return values

def _write_dict_to_file(repository_ctx, filepath, header, values):
    content = header + "\n"
    for key, value in values.items():
        content += '{} = "{}"\n'.format(key, value)

    content += "\nllvm_vars = {\n"
    for key, value in values.items():
        content += '    "{}": "{}",\n'.format(key, value)
    content += "}\n"

    repository_ctx.file(filepath, content = content)

def _llvm_configure_impl(repository_ctx):
    _overlay_directories(repository_ctx)

    vars = _extract_cmake_settings(repository_ctx, "llvm/CMakeLists.txt")
    version = _extract_cmake_settings(repository_ctx, "cmake/Modules/LLVMVersion.cmake")
    vars.update({key: value for key, value in version.items() if value != None})

    _write_dict_to_file(
        repository_ctx,
        filepath = "vars.bzl",
        header = "# Generated from llvm/CMakeLists.txt\n",
        values = vars,
    )

    repository_ctx.file(
        "llvm/targets.bzl",
        content = "llvm_targets = " + str(repository_ctx.attr.targets),
        executable = False,
    )

    bolt_targets = [target for target in repository_ctx.attr.targets if target in ["AArch64", "RISCV", "X86"]]
    repository_ctx.file(
        "bolt/targets.bzl",
        content = "bolt_targets = " + str(bolt_targets),
        executable = False,
    )

llvm_configure = repository_rule(
    implementation = _llvm_configure_impl,
    local = True,
    configure = True,
    attrs = {
        "llvm_raw_repo": attr.string(default = "llvm-raw"),
        "targets": attr.string_list(default = DEFAULT_TARGETS),
    },
)
