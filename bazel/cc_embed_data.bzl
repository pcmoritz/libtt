load("@rules_cc//cc:defs.bzl", "cc_library")

_ASM_TEMPLATE = Label("@//:bazel/cc_embed_data.S.tpl")
_HEADER_TEMPLATE = Label("@//:bazel/cc_embed_data.h.tpl")

def _cc_embed_data_src_impl(ctx):
    data = ctx.file.src
    asm = ctx.actions.declare_file(ctx.attr.out_prefix + ".S")
    hdr = ctx.actions.declare_file(ctx.attr.out_prefix + ".h")
    fingerprint = ctx.actions.declare_file(ctx.attr.out_prefix + ".sha256")

    substitutions = {
        "@DATA_SYMBOL@": ctx.attr.data_symbol,
        "@FINGERPRINT_SYMBOL@": ctx.attr.fingerprint_symbol,
        "@NAMESPACE_CLOSE@": "}  // namespace %s" % ctx.attr.namespace,
        "@NAMESPACE_OPEN@": "namespace %s {" % ctx.attr.namespace,
        "@SIZE_SYMBOL@": ctx.attr.size_symbol,
    }

    ctx.actions.run_shell(
        inputs = [data],
        outputs = [fingerprint],
        arguments = [data.path, fingerprint.path],
        command = """
set -eu
fingerprint=$(sha256sum "$1" | cut -d ' ' -f 1)
printf '%s' "$fingerprint" > "$2"
""",
    )
    ctx.actions.expand_template(
        template = ctx.file._header_template,
        output = hdr,
        substitutions = substitutions,
    )

    asm_substitutions = dict(substitutions)
    asm_substitutions.update({
        "@DATA_PATH@": data.path,
        "@FINGERPRINT_PATH@": fingerprint.path,
    })
    ctx.actions.expand_template(
        template = ctx.file._asm_template,
        output = asm,
        substitutions = asm_substitutions,
    )

    return [
        DefaultInfo(files = depset([asm, hdr, fingerprint])),
        OutputGroupInfo(
            asm = depset([asm]),
            fingerprint = depset([fingerprint]),
            hdr = depset([hdr]),
        ),
    ]

_cc_embed_data_src = rule(
    implementation = _cc_embed_data_src_impl,
    attrs = {
        "data_symbol": attr.string(mandatory = True),
        "fingerprint_symbol": attr.string(mandatory = True),
        "namespace": attr.string(mandatory = True),
        "out_prefix": attr.string(mandatory = True),
        "size_symbol": attr.string(mandatory = True),
        "src": attr.label(allow_single_file = True, mandatory = True),
        "_asm_template": attr.label(
            allow_single_file = True,
            default = _ASM_TEMPLATE,
        ),
        "_header_template": attr.label(
            allow_single_file = True,
            default = _HEADER_TEMPLATE,
        ),
    },
)

def cc_embed_data(name, src, namespace, data_symbol, size_symbol, fingerprint_symbol, out_prefix):
    """Embeds a single data file in a C++ library using assembler .incbin."""
    generated = name + "_src"

    _cc_embed_data_src(
        name = generated,
        src = src,
        data_symbol = data_symbol,
        fingerprint_symbol = fingerprint_symbol,
        namespace = namespace,
        out_prefix = out_prefix,
        size_symbol = size_symbol,
    )
    native.filegroup(
        name = name + "_asm",
        srcs = [":" + generated],
        output_group = "asm",
        visibility = ["//visibility:private"],
    )
    native.filegroup(
        name = name + "_fingerprint",
        srcs = [":" + generated],
        output_group = "fingerprint",
        visibility = ["//visibility:private"],
    )
    native.filegroup(
        name = name + "_hdr",
        srcs = [":" + generated],
        output_group = "hdr",
        visibility = ["//visibility:private"],
    )

    cc_library(
        name = name,
        srcs = [":" + name + "_asm"],
        additional_compiler_inputs = [
            src,
            ":" + name + "_fingerprint",
        ],
        hdrs = [":" + name + "_hdr"],
        includes = ["."],
        linkstatic = True,
    )
