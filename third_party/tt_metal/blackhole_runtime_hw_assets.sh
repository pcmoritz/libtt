#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "usage: $0 MAIN_LD RISCV_GPP RULEDIR PROC_ITEMS KINDS OBJ:SOURCE..." >&2
  exit 2
fi

main_ld=$1
gpp=$2
rule_dir=$3
proc_items=$4
kinds=$5
shift 5

root=${main_ld%/tt_metal/hw/toolchain/main.ld}
toolchain_out=$rule_dir/runtime/hw/toolchain/blackhole
lib_out=$rule_dir/runtime/hw/lib/blackhole
mkdir -p "$toolchain_out" "$lib_out"

for item in $proc_items; do
  proc_rest=${item#*:}
  proc_file=${proc_rest%%:*}
  proc_define_upper=${proc_rest##*:}
  for kind in $kinds; do
    kind_name=${kind%%:*}
    kind_upper=${kind##*:}
    "$gpp" -DTYPE_"$kind_upper" -DCOMPILE_FOR_"$proc_define_upper" -DARCH_BLACKHOLE \
      -I"$root/tt_metal/hw/inc/internal/tt-1xx/blackhole" \
      -E -P -x c -o "$toolchain_out/${kind_name}_${proc_file}.ld" \
      "$root/tt_metal/hw/toolchain/main.ld"
  done
done

common_flags=(
  -mcpu=tt-bh
  -std=c++17
  -flto=auto
  -ffast-math
  -fno-use-cxa-atexit
  -fno-exceptions
  -Wall
  -Werror
  -Wno-deprecated-declarations
  -Wno-unknown-pragmas
  -Wno-error=multistatement-macros
  -Wno-error=parentheses
  -Wno-error=unused-but-set-variable
  -Wno-unused-variable
  -Wno-unused-function
  -Os
  -fno-tree-loop-distribute-patterns
)

includes=(
  -I.
  -I..
  -I"$root"
  -I"$root/tt_metal"
  -I"$root/tt_metal/api"
  -I"$root/tt_metal/api/tt-metalium"
  -I"$root/tt_metal/hw/inc"
  -I"$root/tt_metal/hw/inc/debug"
  -I"$root/tt_metal/hw/firmware/src/tt-1xx"
  -I"$root/tt_metal/hw/inc/internal/tt-1xx/blackhole"
  -I"$root/tt_metal/hw/inc/internal/tt-1xx/blackhole/blackhole_defines"
  -I"$root/tt_metal/hw/inc/internal/tt-1xx/blackhole/noc"
  -I"$root/tt_metal/hw/inc/internal/tt-1xx/blackhole/overlay"
  -I"$root/tt_metal/third_party/umd/device/blackhole"
  -I"$root/tt_metal/hw/ckernels/blackhole/metal/common"
  -I"$root/tt_metal/hw/ckernels/blackhole/metal/llk_io"
  -I"$root/tt_metal/tt-llk/tt_llk_blackhole/common/inc"
  -I"$root/tt_metal/tt-llk/tt_llk_blackhole/llk_lib"
)

for obj_src in "$@"; do
  obj=${obj_src%%:*}
  src=${obj_src#*:}
  "$gpp" "${common_flags[@]}" -DTENSIX_FIRMWARE "${includes[@]}" \
    -c -o "$lib_out/$obj" "$root/$src"
done
