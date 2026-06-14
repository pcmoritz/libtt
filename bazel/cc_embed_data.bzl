load("@rules_cc//cc:defs.bzl", "cc_library")

_CC_EMBED_DATA_CMD = """
set -eu
data="$(location {src})"
fingerprint=$$(sha256sum "$$data" | cut -d ' ' -f 1)

cat > "$(@D)/{hdr}" <<'EOF'
#pragma once

#include <cstddef>

{namespace_open}

{extern_c_open}

extern const unsigned char {data_symbol}[];
extern const std::size_t {size_symbol};
extern const char {fingerprint_symbol}[];

{extern_c_close}

{namespace_close}
EOF

cat > "$(@D)/{asm}" <<EOF
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
"""

def cc_embed_data(name, src, namespace, data_symbol, size_symbol, fingerprint_symbol, out_prefix):
    """Embeds a single data file in a C++ library using assembler .incbin."""
    asm = out_prefix + ".S"
    hdr = out_prefix + ".h"
    cmd = _CC_EMBED_DATA_CMD.format(
        asm = asm,
        data_symbol = data_symbol,
        extern_c_close = "}  // extern \"C\"",
        extern_c_open = "extern \"C\" {",
        fingerprint_symbol = fingerprint_symbol,
        hdr = hdr,
        namespace_close = "}  // namespace %s" % namespace,
        namespace_open = "namespace %s {" % namespace,
        size_symbol = size_symbol,
        src = src,
    )

    native.genrule(
        name = name + "_src",
        srcs = [src],
        outs = [asm, hdr],
        cmd = cmd,
    )

    cc_library(
        name = name,
        srcs = [asm],
        additional_compiler_inputs = [src],
        hdrs = [hdr],
        includes = ["."],
        linkstatic = True,
    )
