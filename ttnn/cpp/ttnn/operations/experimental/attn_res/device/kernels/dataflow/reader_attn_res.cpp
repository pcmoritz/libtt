// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// This core's tile columns of the queries, every source, the addend and the
// weight, in the order the compute kernel takes them.
void kernel_main() {
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    const uint32_t queries_addr = get_arg_val<uint32_t>(2);
    const uint32_t addend_addr = get_arg_val<uint32_t>(3);
    const uint32_t weight_addr = get_arg_val<uint32_t>(4);
    constexpr uint32_t sources_arg = 5;

    constexpr uint32_t cb_sources = get_named_compile_time_arg_val("cb_sources");
    constexpr uint32_t cb_addend = get_named_compile_time_arg_val("cb_addend");
    constexpr uint32_t cb_queries = get_named_compile_time_arg_val("cb_queries");
    constexpr uint32_t cb_weight = get_named_compile_time_arg_val("cb_weight");
    constexpr uint32_t num_sources = get_named_compile_time_arg_val("num_sources");
    constexpr bool has_addend = get_named_compile_time_arg_val("has_addend") != 0;
    constexpr bool has_weight = get_named_compile_time_arg_val("has_weight") != 0;
    constexpr auto source_args = TensorAccessorArgs<0>();
    constexpr auto addend_args = TensorAccessorArgs<source_args.next_compile_time_args_offset()>();
    constexpr auto queries_args = TensorAccessorArgs<addend_args.next_compile_time_args_offset()>();
    constexpr auto weight_args = TensorAccessorArgs<queries_args.next_compile_time_args_offset()>();
    const uint32_t source_tile = get_tile_size(cb_sources);

    auto read = [&](const auto& tensor, uint32_t cb, uint32_t tile_bytes) {
        cb_reserve_back(cb, count);
        const uint32_t dest = get_write_ptr(cb);
        for (uint32_t c = 0; c < count; ++c) {
            noc_async_read(tensor.get_noc_addr(start + c), dest + c * tile_bytes, tile_bytes);
        }
    };

    const auto queries = TensorAccessor(queries_args, queries_addr, get_tile_size(cb_queries));
    read(queries, cb_queries, get_tile_size(cb_queries));
    // Every source's tiles, source by source.
    cb_reserve_back(cb_sources, num_sources * count);
    const uint32_t sources_dest = get_write_ptr(cb_sources);
    for (uint32_t j = 0; j < num_sources; ++j) {
        const auto source = TensorAccessor(source_args, get_arg_val<uint32_t>(sources_arg + j), source_tile);
        for (uint32_t c = 0; c < count; ++c) {
            noc_async_read(
                source.get_noc_addr(start + c), sources_dest + (j * count + c) * source_tile, source_tile);
        }
    }
    if constexpr (has_addend) {
        const auto addend = TensorAccessor(addend_args, addend_addr, get_tile_size(cb_addend));
        read(addend, cb_addend, get_tile_size(cb_addend));
    }
    noc_async_read_barrier();
    cb_push_back(cb_queries, count);
    cb_push_back(cb_sources, num_sources * count);
    if constexpr (has_addend) {
        cb_push_back(cb_addend, count);
    }
    if constexpr (has_weight) {
        const auto weight = TensorAccessor(weight_args, weight_addr, get_tile_size(cb_weight));
        read(weight, cb_weight, get_tile_size(cb_weight));
        noc_async_read_barrier();
        cb_push_back(cb_weight, count);
    }
}
