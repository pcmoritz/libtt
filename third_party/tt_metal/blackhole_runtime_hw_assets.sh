#!/usr/bin/env bash
set -euo pipefail

main_ld=$1
dev_mem_map=$2
gpp=$3
out_dir=$4
shift 4

mkdir -p "$out_dir"
include_dir=$(dirname "$dev_mem_map")

while [ "$#" -gt 0 ]; do
  out=$1
  kind_define=$2
  proc_define=$3
  shift 3

  "$gpp" -DTYPE_"$kind_define" -DCOMPILE_FOR_"$proc_define" -DARCH_BLACKHOLE \
    -I"$include_dir" -E -P -x c \
    -o "$out_dir/$out" "$main_ld"
done
