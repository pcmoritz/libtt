#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 MAIN_LD RISCV_GPP RULEDIR PROC_ITEMS KINDS" >&2
  exit 2
fi

main_ld=$1
gpp=$2
rule_dir=$3
proc_items=$4
kinds=$5

root=${main_ld%/tt_metal/hw/toolchain/main.ld}
toolchain_out=$rule_dir/runtime/hw/toolchain/blackhole
mkdir -p "$toolchain_out"

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
