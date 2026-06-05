use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer, TILE_C, TILE_R};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{Kernel, RuntimeArgsBuilder};
use std::io;

const READER: &str = include_str!("../../kernels/sdpa_decode_reader.cc");
const WRITER: &str = include_str!("../../kernels/sdpa_decode_writer.cc");
const COMPUTE_COMMON: &str = include_str!("../../kernels/sdpa_decode_compute_common.cc.inc");
const COMPUTE_TEMPLATE: &str = include_str!("../../kernels/sdpa_decode_compute.cc");

const READER_Q_ADDR_INDEX: usize = 0;
const READER_K_ADDR_INDEX: usize = 1;
const READER_V_ADDR_INDEX: usize = 2;
const READER_SEQ_LENS_ADDR_INDEX: usize = 3;
const READER_LOC_ADDR_INDEX: usize = 4;
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct SdpaDecodeProgramKey {
    cores: Vec<CoreCoord>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    cache_tokens: usize,
    key_tokens: usize,
    scale_bf16_packed: u32,
}

struct SdpaDecodeKernel {
    q_addr: u32,
    k_addr: u32,
    v_addr: u32,
    seq_lens_addr: u32,
    loc_addr: u32,
    output_addr: u32,
    key: SdpaDecodeProgramKey,
}

impl Kernel<SdpaDecodeProgramKey> for SdpaDecodeKernel {
    fn program_key(&self) -> SdpaDecodeProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        sdpa_decode_program(self.key.clone())
    }

    fn reader_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            READER_Q_ADDR_INDEX => Some(self.q_addr),
            READER_K_ADDR_INDEX => Some(self.k_addr),
            READER_V_ADDR_INDEX => Some(self.v_addr),
            READER_SEQ_LENS_ADDR_INDEX => Some(self.seq_lens_addr),
            READER_LOC_ADDR_INDEX => Some(self.loc_addr),
            _ => None,
        }
    }

    fn writer_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        (index == WRITER_OUTPUT_ADDR_INDEX).then_some(self.output_addr)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sdpa_decode(
    device: &mut Device,
    q: &DramBuffer,
    k: &DramBuffer,
    v: &DramBuffer,
    seq_lens: &DramBuffer,
    loc: &DramBuffer,
    q_shape: &[usize],
    k_shape: &[usize],
    v_shape: &[usize],
    seq_lens_shape: &[usize],
    loc_shape: &[usize],
    output_shape: &[usize],
    scale_bf16_packed: u32,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    let shape = validate_sdpa_decode_shapes(
        q, k, v, seq_lens, loc, q_shape, k_shape, v_shape, seq_lens_shape, loc_shape,
        output_shape,
    )?;
    let output_allocation_shape = tiled_allocation_shape(output_shape)?;
    let output_tiles = tiled_shape_tile_count(output_shape)?;
    let output = device.alloc(output_tiles, DType::Float16B, &output_allocation_shape, name)?;
    let cores = select_kv_head_cores(device.cores_ref(), shape.kv_heads)?;

    let key = SdpaDecodeProgramKey {
        cores,
        q_heads: shape.q_heads,
        kv_heads: shape.kv_heads,
        head_dim: shape.head_dim,
        cache_tokens: shape.cache_tokens,
        key_tokens: shape.key_tokens,
        scale_bf16_packed,
    };
    let kernel = SdpaDecodeKernel {
        q_addr: u32_addr(q.addr, "q address")?,
        k_addr: u32_addr(k.addr, "k address")?,
        v_addr: u32_addr(v.addr, "v address")?,
        seq_lens_addr: u32_addr(seq_lens.addr, "seq_lens address")?,
        loc_addr: u32_addr(loc.addr, "loc address")?,
        output_addr: u32_addr(output.addr, "sdpa output address")?,
        key,
    };
    kernel.run(device)?;
    Ok(output)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SdpaDecodeShape {
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    cache_tokens: usize,
    key_tokens: usize,
}

#[allow(clippy::too_many_arguments)]
fn validate_sdpa_decode_shapes(
    q: &DramBuffer,
    k: &DramBuffer,
    v: &DramBuffer,
    seq_lens: &DramBuffer,
    loc: &DramBuffer,
    q_shape: &[usize],
    k_shape: &[usize],
    v_shape: &[usize],
    seq_lens_shape: &[usize],
    loc_shape: &[usize],
    output_shape: &[usize],
) -> io::Result<SdpaDecodeShape> {
    if q.dtype != DType::Float16B || k.dtype != DType::Float16B || v.dtype != DType::Float16B {
        return Err(invalid_input(format!(
            "sdpa_decode requires bf16 q/k/v, got {:?}/{:?}/{:?}",
            q.dtype, k.dtype, v.dtype
        )));
    }
    if seq_lens.dtype != DType::Int32 || loc.dtype != DType::Int32 {
        return Err(invalid_input(format!(
            "sdpa_decode requires s32 seq_lens/loc, got {:?}/{:?}",
            seq_lens.dtype, loc.dtype
        )));
    }
    if q_shape.len() != 3 || k_shape.len() != 3 || v_shape.len() != 3 || output_shape.len() != 3 {
        return Err(invalid_input(format!(
            "sdpa_decode requires rank-3 q/k/v/output, got q={q_shape:?} k={k_shape:?} v={v_shape:?} output={output_shape:?}"
        )));
    }
    if seq_lens_shape != [1] || loc_shape.len() != 1 {
        return Err(invalid_input(format!(
            "sdpa_decode requires seq_lens=[1] and rank-1 loc, got seq_lens={seq_lens_shape:?} loc={loc_shape:?}"
        )));
    }
    if k_shape != v_shape {
        return Err(invalid_input(format!(
            "sdpa_decode k/v shape mismatch: {k_shape:?} vs {v_shape:?}"
        )));
    }
    let [q_batch, q_heads, head_dim]: [usize; 3] = q_shape.try_into().expect("rank checked");
    let [cache_tokens, kv_heads, kv_head_dim]: [usize; 3] = k_shape.try_into().expect("rank checked");
    if q_batch != 1 || output_shape != [1, q_heads, head_dim] {
        return Err(invalid_input(format!(
            "sdpa_decode currently requires q batch 1 and output [1, q_heads, head_dim], got q={q_shape:?} output={output_shape:?}"
        )));
    }
    if q_heads != TILE_R {
        return Err(invalid_input(format!(
            "sdpa_decode currently supports exactly {TILE_R} query heads, got {q_heads}"
        )));
    }
    if kv_heads == 0 || q_heads % kv_heads != 0 || kv_heads > TILE_R {
        return Err(invalid_input(format!(
            "sdpa_decode requires 1..={TILE_R} kv heads dividing q heads, got q_heads={q_heads} kv_heads={kv_heads}"
        )));
    }
    if head_dim != kv_head_dim || head_dim == 0 || head_dim % TILE_C != 0 {
        return Err(invalid_input(format!(
            "sdpa_decode requires matching head_dim divisible by {TILE_C}, got q={head_dim} kv={kv_head_dim}"
        )));
    }
    let key_tokens = loc_shape[0];
    if key_tokens == 0 || key_tokens % TILE_R != 0 || cache_tokens == 0 {
        return Err(invalid_input(format!(
            "sdpa_decode requires non-empty tiled key/cache lengths, got key_tokens={key_tokens} cache_tokens={cache_tokens}"
        )));
    }
    validate_tiled_buffer(q, q_shape, "q")?;
    validate_tiled_buffer(k, k_shape, "k")?;
    validate_tiled_buffer(v, v_shape, "v")?;
    validate_tiled_buffer(seq_lens, seq_lens_shape, "seq_lens")?;
    validate_tiled_buffer(loc, loc_shape, "loc")?;
    Ok(SdpaDecodeShape {
        q_heads,
        kv_heads,
        head_dim,
        cache_tokens,
        key_tokens,
    })
}

fn validate_tiled_buffer(buffer: &DramBuffer, logical_shape: &[usize], name: &str) -> io::Result<()> {
    let expected_shape = tiled_allocation_shape(logical_shape)?;
    let expected_tiles = tiled_shape_tile_count(logical_shape)?;
    if buffer.shape != expected_shape || buffer.num_tiles != expected_tiles {
        return Err(invalid_input(format!(
            "sdpa_decode {name} allocation mismatch: got shape {:?} tiles {}, expected shape {:?} tiles {}",
            buffer.shape, buffer.num_tiles, expected_shape, expected_tiles
        )));
    }
    Ok(())
}

fn select_kv_head_cores(available: &[CoreCoord], kv_heads: usize) -> io::Result<Vec<CoreCoord>> {
    if available.len() < kv_heads {
        return Err(invalid_input(format!(
            "sdpa_decode requires at least {kv_heads} worker cores, got {}",
            available.len()
        )));
    }
    Ok(available[..kv_heads].to_vec())
}

fn sdpa_decode_program(key: SdpaDecodeProgramKey) -> io::Result<Program> {
    let dht = key.head_dim / TILE_C;
    let st = key.key_tokens / TILE_R;
    let sk_chunk_t = sdpa_key_chunk_tiles(st);
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        vec![WRITER_OUTPUT_ADDR_INDEX],
        vec![
            READER_Q_ADDR_INDEX,
            READER_K_ADDR_INDEX,
            READER_V_ADDR_INDEX,
            READER_SEQ_LENS_ADDR_INDEX,
            READER_LOC_ADDR_INDEX,
        ],
        Vec::new(),
    );
    for (kv_head, &core) in key.cores.iter().enumerate() {
        runtime_args.add_core(
            core,
            vec![0, u32_arg(kv_head, "kv head")?],
            vec![0, 0, 0, 0, 0, u32_arg(kv_head, "kv head")?],
            vec![
                1,
                1,
                u32_arg(kv_head, "kv head")?,
                0,
                0,
                u32_arg(kv_head, "kv head")?,
                u32::MAX,
            ],
        )?;
    }
    let runtime_args = runtime_args.build()?;
    let q_tiles = dht;
    let kv_tiles = sk_chunk_t * dht;
    let qk_tiles = sk_chunk_t;
    let out_tiles = dht;
    Ok(Program {
        reader_kernel: reader_source(&key, st, dht, sk_chunk_t),
        compute_kernel: compute_source(&key, st, dht, sk_chunk_t),
        writer_kernel: writer_source(&key, dht),
        compile: CompileConfig {
            cbs: vec![
                CBConfig::new(0, DType::Float16B).with_tiles(q_tiles),
                CBConfig::new(1, DType::Float16B).with_tiles(2 * kv_tiles),
                CBConfig::new(2, DType::Float16B).with_tiles(2 * kv_tiles),
                CBConfig::new(3, DType::Float16B).with_tiles(qk_tiles),
                CBConfig::new(4, DType::Float16B).with_tiles(1),
                CBConfig::new(5, DType::Float16B).with_tiles(1),
                CBConfig::new(6, DType::Float16B).with_tiles(1),
                CBConfig::new(7, DType::Float16B).with_tiles(1),
                CBConfig::new(8, DType::Int32).with_tiles(1),
                CBConfig::new(9, DType::Int32).with_tiles(1),
                CBConfig::new(10, DType::Float16B).with_tiles(q_tiles.max(2)),
                CBConfig::new(16, DType::Float16B).with_tiles(out_tiles),
                CBConfig::new(17, DType::Float16B).with_tiles(1),
                CBConfig::new(18, DType::Float16B).with_tiles(1),
                CBConfig::new(20, DType::Float16B).with_tiles(out_tiles),
                CBConfig::new(21, DType::Float16B).with_tiles(1),
                CBConfig::new(22, DType::Float16B).with_tiles(1),
                CBConfig::new(23, DType::Float16B).with_tiles(out_tiles),
                CBConfig::new(24, DType::Float16B).with_tiles(qk_tiles),
                CBConfig::new(25, DType::Float16B).with_tiles(out_tiles),
                CBConfig::new(26, DType::Float16B).with_tiles(out_tiles),
                CBConfig::new(27, DType::Float16B).with_tiles(1),
                CBConfig::new(28, DType::Float16B).with_tiles(1),
                CBConfig::new(29, DType::Float16B).with_tiles(1),
                CBConfig::new(30, DType::Float16B).with_tiles(1),
                CBConfig::new(31, DType::Float16B).with_tiles(1),
            ],
            dst_accum_mode: true,
            ..CompileConfig::default()
        },
        name: format!(
            "sdpa_decode_q{}_kv{}_d{}_s{}",
            key.q_heads, key.kv_heads, key.head_dim, key.key_tokens
        ),
        ..Program::new(runtime_args)
    })
}

fn sdpa_key_chunk_tiles(st: usize) -> usize {
    if st <= 8 {
        st
    } else if st % 4 == 0 {
        4
    } else if st % 2 == 0 {
        2
    } else {
        1
    }
}

fn reader_source(key: &SdpaDecodeProgramKey, st: usize, dht: usize, sk_chunk_t: usize) -> String {
    format!(
        "#define SDPA_ST {st}\n#define SDPA_DHT {dht}\n#define SDPA_SK_CHUNK_T {sk_chunk_t}\n#define SDPA_KV_HEADS {}\n#define SDPA_CACHE_TOKENS {}\n{}",
        key.kv_heads, key.cache_tokens, READER
    )
}

fn writer_source(key: &SdpaDecodeProgramKey, dht: usize) -> String {
    format!(
        "#define SDPA_DHT {dht}\n#define SDPA_Q_HEADS {}\n#define SDPA_KV_HEADS {}\n#define SDPA_SCALE_BF16_PACKED 0x{:08x}u\n{}",
        key.q_heads, key.kv_heads, key.scale_bf16_packed, WRITER
    )
}

fn compute_source(key: &SdpaDecodeProgramKey, st: usize, dht: usize, sk_chunk_t: usize) -> String {
    let pnh_t = key.q_heads / TILE_R;
    let dst_size = 8usize;
    let qk_in0_block_w = dht;
    let qk_subblock_w = sk_chunk_t.min(dst_size);
    let qk_subblock_h = if qk_subblock_w == sk_chunk_t {
        pnh_t.min(dst_size / qk_subblock_w).max(1)
    } else {
        1
    };
    let qk_in0_num_subblocks = pnh_t / qk_subblock_h;
    let qk_in1_num_subblocks = sk_chunk_t / qk_subblock_w;
    let out_in0_block_w = sk_chunk_t;
    let out_subblock_w = dht.min(dst_size);
    let out_subblock_h = if out_subblock_w == dht {
        pnh_t.min(dst_size / out_subblock_w).max(1)
    } else {
        1
    };
    let out_in0_num_subblocks = pnh_t / out_subblock_h;
    let out_in1_num_subblocks = dht / out_subblock_w;
    let mut source = format!(
        "#include <cstdint>\n#define EXP_APPROX_MODE false\n{}\n{}\n{}",
        SDPA_RUNTIME_HELPERS,
        COMPUTE_COMMON,
        COMPUTE_TEMPLATE
    );
    source = source.replace(
        "#include \"cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp\"\n#include \"compute_common.hpp\"\n",
        "",
    );
    source = source.replace("tt::constants::TILE_HEIGHT", "32");
    let replacements = [
        (0, st),
        (1, dht),
        (2, pnh_t),
        (3, sk_chunk_t),
        (4, qk_in0_block_w),
        (5, qk_subblock_w),
        (6, qk_subblock_h),
        (7, qk_in0_num_subblocks),
        (8, qk_in1_num_subblocks),
        (9, 1),
        (10, out_in0_block_w),
        (11, out_subblock_w),
        (12, out_subblock_h),
        (13, out_in0_num_subblocks),
        (14, out_in1_num_subblocks),
        (15, 1),
        (18, 1),
        (19, 1),
        (20, 0),
        (21, 1),
        (22, sk_chunk_t),
        (23, 0),
    ];
    for (index, value) in replacements {
        source = source.replace(
            &format!("get_compile_time_arg_val({index})"),
            &value.to_string(),
        );
    }
    source
}

const SDPA_RUNTIME_HELPERS: &str = r#"
struct SdpaRuntimeArgs {
  uint32_t pst;
  uint32_t num_chunks;
  uint32_t chunk_start;
  uint32_t chunk_end;
};

inline uint32_t nearest_n(uint32_t x, uint32_t n) {
  return ((x + n - 1) / n) * n;
}

inline uint32_t min_u32(uint32_t lhs, uint32_t rhs) {
  return lhs < rhs ? lhs : rhs;
}

inline SdpaRuntimeArgs get_runtime_args(
    int cur_pos, int, int core_num, int num_cores_per_batch, uint32_t k_chunk_size) {
  uint32_t valid_seq_len = nearest_n(cur_pos + 1, k_chunk_size);
  uint32_t pst_value = valid_seq_len / 32;
  uint32_t num_chunks_value = valid_seq_len / k_chunk_size;
  uint32_t k_chunk_start = 0;
  uint32_t k_chunk_end = 0;

  if (static_cast<uint32_t>(num_cores_per_batch) > num_chunks_value) {
    uint32_t chunks_per_core = static_cast<uint32_t>(core_num) < num_chunks_value ? 1 : 0;
    k_chunk_start = (num_chunks_value - static_cast<uint32_t>(core_num) - 1) * chunks_per_core;
    k_chunk_end = (num_chunks_value - static_cast<uint32_t>(core_num)) * chunks_per_core;
  } else {
    uint32_t chunks_per_core = num_chunks_value / static_cast<uint32_t>(num_cores_per_batch);
    uint32_t residuals = num_chunks_value % static_cast<uint32_t>(num_cores_per_batch);
    uint32_t reversed_core_num =
        static_cast<uint32_t>(num_cores_per_batch) - static_cast<uint32_t>(core_num) - 1;
    k_chunk_start = reversed_core_num * chunks_per_core + min_u32(residuals, reversed_core_num);
    k_chunk_end = k_chunk_start + chunks_per_core;
    if (reversed_core_num < residuals) {
      k_chunk_end += 1;
    }
  }
  return {pst_value, num_chunks_value, k_chunk_start, k_chunk_end};
}

template <uint32_t Sk_chunk_t, uint32_t>
inline uint32_t get_dynamic_Sk_chunk_t(int) {
  return Sk_chunk_t;
}
"#;

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn u32_arg(value: usize, name: &str) -> io::Result<u32> {
    u32::try_from(value)
        .map_err(|_| invalid_input(format!("{name} does not fit in u32: 0x{value:x}")))
}

fn u32_addr(value: u64, name: &str) -> io::Result<u32> {
    u32::try_from(value)
        .map_err(|_| invalid_input(format!("{name} does not fit in u32: 0x{value:x}")))
}
