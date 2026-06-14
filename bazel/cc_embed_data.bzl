load("@rules_cc//cc:defs.bzl", "cc_library")

def cc_embed_data(
        name,
        src,
        namespace,
        data_symbol,
        size_symbol,
        fingerprint_symbol,
        out_prefix = None,
        visibility = None,
        includes = None,
        testonly = None,
        tags = None):
    """Embeds a single data file in a C++ library using assembler .incbin."""
    if out_prefix == None:
        out_prefix = name

    asm = out_prefix + ".S"
    hdr = out_prefix + ".h"
    namespace_open = "namespace %s {" % namespace
    namespace_close = "}  // namespace %s" % namespace

    genrule_kwargs = {}
    cc_library_kwargs = {}
    if testonly != None:
        genrule_kwargs["testonly"] = testonly
        cc_library_kwargs["testonly"] = testonly
    if tags != None:
        genrule_kwargs["tags"] = tags
        cc_library_kwargs["tags"] = tags
    if includes != None:
        cc_library_kwargs["includes"] = includes
    if visibility != None:
        cc_library_kwargs["visibility"] = visibility

    native.genrule(
        name = name + "_src",
        srcs = [src],
        outs = [
            asm,
            hdr,
        ],
        cmd = """
set -eu
data="$(location {src})"
fingerprint=$$(sha256sum "$$data" | cut -d ' ' -f 1)
asm="$(@D)/{asm}"
hdr="$(@D)/{hdr}"

cat > "$$hdr" <<'EOF'
#pragma once

#include <cstddef>

{namespace_open}

extern "C" {{

extern const unsigned char {data_symbol}[];
extern const std::size_t {size_symbol};
extern const char {fingerprint_symbol}[];

}}  // extern "C"

{namespace_close}
EOF

cat > "$$asm" <<EOF
.section .rodata
.balign 16
.global {data_symbol}
{data_symbol}:
  .incbin "$$data"
{data_symbol}End:
.balign 8
.global {size_symbol}
{size_symbol}:
  .quad {data_symbol}End - {data_symbol}
.global {fingerprint_symbol}
{fingerprint_symbol}:
  .asciz "$$fingerprint"
.section .note.GNU-stack,"",@progbits
EOF
""".format(
            asm = asm,
            data_symbol = data_symbol,
            fingerprint_symbol = fingerprint_symbol,
            hdr = hdr,
            namespace_close = namespace_close,
            namespace_open = namespace_open,
            size_symbol = size_symbol,
            src = src,
        ),
        **genrule_kwargs
    )

    cc_library(
        name = name,
        srcs = [asm],
        additional_compiler_inputs = [src],
        hdrs = [hdr],
        linkstatic = True,
        **cc_library_kwargs
    )
