use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer};
use crate::executable::CompareDirection;
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, split_tile_range, Kernel, RuntimeArgsBuilder};
use std::fmt::Display;
use std::io;

const WRITER: &str = include_str!("../../kernels/tile_writer.cc");
const MAX_FUSED_INPUTS: usize = 8;
const MAX_FUSED_NODES: usize = 16;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) enum FusedEltwiseOp {
    Input,
    Constant,
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Max,
    Compare(CompareDirection),
    Cosine,
    Sine,
    Negate,
    Exponential,
    Rsqrt,
    Convert,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct FusedEltwiseNode {
    pub(crate) op: FusedEltwiseOp,
    pub(crate) input_nodes: Vec<u32>,
    pub(crate) input_index: u32,
    pub(crate) packed_value: u32,
    pub(crate) dtype: DType,
    pub(crate) single_tile_broadcast: bool,
}

#[derive(Clone, Copy)]
pub(crate) enum FusedEltwiseInput<'a> {
    Dram {
        buffer: &'a DramBuffer,
        // Logical scalar broadcasts are passed as a one-tile input. The reader
        // replicates the first element across each output tile before compute.
        single_tile_broadcast: bool,
    },
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct FusedEltwiseProgramKey {
    cores: Vec<CoreCoord>,
    tile_count: u32,
    input_dtypes: Vec<DType>,
    input_broadcasts: Vec<bool>,
    output_dtype: DType,
    nodes: Vec<FusedEltwiseNode>,
    root_node_id: u32,
}

struct FusedEltwiseKernel {
    input_addrs: Vec<u32>,
    output_addr: u32,
    key: FusedEltwiseProgramKey,
}

impl Kernel<FusedEltwiseProgramKey> for FusedEltwiseKernel {
    fn program_key(&self) -> FusedEltwiseProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        fused_eltwise_program(self.key.clone())
    }

    #[inline]
    fn reader_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        let input_count = self.input_addrs.len();
        if index < input_count {
            return Some(self.input_addrs[index]);
        }
        None
    }

    #[inline]
    fn writer_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        (index == 0).then_some(self.output_addr)
    }
}

pub(crate) fn eltwise(
    device: &mut Device,
    external_inputs: &[FusedEltwiseInput<'_>],
    nodes: &[FusedEltwiseNode],
    root_node_id: u32,
    shape: &[usize],
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    validate_fused_eltwise(external_inputs, nodes, root_node_id, shape)?;
    let leaf_inputs = leaf_inputs(external_inputs, nodes)?;

    let output_tiles = tiled_shape_tile_count(shape)?;
    let tile_count = u32_arg(output_tiles, "tile count")?;
    let cores = select_worker_cores(device.cores_ref(), output_tiles)?;
    let output_dtype = nodes[root_node_id as usize].dtype;
    let output_shape = tiled_allocation_shape(shape)?;
    let output = device.alloc(output_tiles, output_dtype, &output_shape, name)?;

    let mut input_addrs = Vec::with_capacity(leaf_inputs.len());
    let mut input_dtypes = Vec::with_capacity(leaf_inputs.len());
    for (index, input) in leaf_inputs.iter().enumerate() {
        match input {
            FusedEltwiseInput::Dram { buffer, .. } => {
                input_addrs.push(u32_arg(buffer.addr, &format!("input[{index}] address"))?);
                input_dtypes.push(buffer.dtype);
            }
        }
    }

    let kernel = FusedEltwiseKernel {
        input_addrs,
        output_addr: u32_arg(output.addr, "output address")?,
        key: FusedEltwiseProgramKey {
            cores,
            tile_count,
            input_dtypes,
            input_broadcasts: leaf_inputs
                .iter()
                .map(|input| {
                    let FusedEltwiseInput::Dram {
                        single_tile_broadcast,
                        ..
                    } = input;
                    *single_tile_broadcast
                })
                .collect(),
            output_dtype,
            nodes: nodes.to_vec(),
            root_node_id,
        },
    };
    kernel.run(device)?;
    Ok(output)
}

fn validate_fused_eltwise(
    external_inputs: &[FusedEltwiseInput<'_>],
    nodes: &[FusedEltwiseNode],
    root_node_id: u32,
    shape: &[usize],
) -> io::Result<()> {
    if external_inputs.len() > MAX_FUSED_INPUTS {
        return Err(invalid_input(format!(
            "fused eltwise supports at most {MAX_FUSED_INPUTS} external inputs, got {}",
            external_inputs.len()
        )));
    }
    if nodes.is_empty() || nodes.len() > MAX_FUSED_NODES {
        return Err(invalid_input(format!(
            "fused eltwise requires 1..={MAX_FUSED_NODES} nodes, got {}",
            nodes.len()
        )));
    }
    let root_index = usize::try_from(root_node_id)
        .map_err(|_| invalid_input(format!("root node id is out of range: {root_node_id}")))?;
    if root_index >= nodes.len() {
        return Err(invalid_input(format!(
            "root node id {root_node_id} is out of bounds for {} nodes",
            nodes.len()
        )));
    }

    let expected_tiles = tiled_shape_tile_count(shape)?;
    let expected_shape = tiled_allocation_shape(shape)?;
    for (index, node) in nodes.iter().enumerate() {
        match node.op {
            FusedEltwiseOp::Input => {
                if !is_supported_leaf_dtype(node.dtype) {
                    return Err(invalid_input(format!(
                        "node[{index}] input dtype {:?} is not supported by fused eltwise",
                        node.dtype
                    )));
                }
                let input_index = usize::try_from(node.input_index).map_err(|_| {
                    invalid_input(format!("node[{index}] input index is out of range"))
                })?;
                if input_index >= external_inputs.len() {
                    return Err(invalid_input(format!(
                        "node[{index}] input index {} is out of bounds for {} inputs",
                        node.input_index,
                        external_inputs.len()
                    )));
                }
                let FusedEltwiseInput::Dram { buffer, .. } = external_inputs[input_index];
                let input_dtype = buffer.dtype;
                if input_dtype != node.dtype {
                    return Err(invalid_input(format!(
                        "node[{index}] input dtype mismatch: node {:?}, input {:?}",
                        node.dtype, input_dtype
                    )));
                }
                if !node.input_nodes.is_empty() {
                    return Err(invalid_input(format!(
                        "node[{index}] input node must not have operands"
                    )));
                }
                if node.single_tile_broadcast {
                    if buffer.num_tiles != 1 {
                        return Err(invalid_input(format!(
                            "node[{index}] single-tile broadcast input has {} tiles, expected 1",
                            buffer.num_tiles
                        )));
                    }
                } else {
                    if buffer.shape != expected_shape {
                        return Err(invalid_input(format!(
                            "node[{index}] input allocation shape mismatch: got {:?}, expected {:?} for logical shape {:?}",
                            buffer.shape, expected_shape, shape
                        )));
                    }
                    if buffer.num_tiles != expected_tiles {
                        return Err(invalid_input(format!(
                            "node[{index}] input tile count mismatch: got {}, expected {expected_tiles}",
                            buffer.num_tiles
                        )));
                    }
                }
            }
            FusedEltwiseOp::Constant => {
                if !is_supported_leaf_dtype(node.dtype) {
                    return Err(invalid_input(format!(
                        "node[{index}] constant dtype {:?} is not supported by fused eltwise",
                        node.dtype
                    )));
                }
                if !node.input_nodes.is_empty() {
                    return Err(invalid_input(format!(
                        "node[{index}] constant node must not have operands"
                    )));
                }
            }
            FusedEltwiseOp::Negate
            | FusedEltwiseOp::Cosine
            | FusedEltwiseOp::Sine
            | FusedEltwiseOp::Exponential
            | FusedEltwiseOp::Rsqrt
            | FusedEltwiseOp::Convert => validate_node_inputs(index, node, 1)?,
            FusedEltwiseOp::Add
            | FusedEltwiseOp::Subtract
            | FusedEltwiseOp::Multiply
            | FusedEltwiseOp::Divide
            | FusedEltwiseOp::Power
            | FusedEltwiseOp::Max
            | FusedEltwiseOp::Compare(_) => validate_node_inputs(index, node, 2)?,
        }
        for &input_node in &node.input_nodes {
            if usize::try_from(input_node).map_or(true, |input| input >= index) {
                return Err(invalid_input(format!(
                    "node[{index}] references non-prior input node {input_node}"
                )));
            }
        }
        validate_node_dtype(index, node, nodes)?;
    }
    let leaf_count = nodes
        .iter()
        .filter(|node| node.op == FusedEltwiseOp::Input)
        .count();
    if leaf_count == 0 || leaf_count > MAX_FUSED_INPUTS {
        return Err(invalid_input(format!(
            "fused eltwise requires 1..={MAX_FUSED_INPUTS} leaf inputs, got {leaf_count}"
        )));
    }
    Ok(())
}

fn leaf_inputs<'a>(
    external_inputs: &[FusedEltwiseInput<'a>],
    nodes: &[FusedEltwiseNode],
) -> io::Result<Vec<FusedEltwiseInput<'a>>> {
    let mut inputs = Vec::new();
    for (index, node) in nodes.iter().enumerate() {
        match node.op {
            FusedEltwiseOp::Input => {
                let input_index = usize::try_from(node.input_index).map_err(|_| {
                    invalid_input(format!("node[{index}] input index is out of range"))
                })?;
                let input = external_inputs.get(input_index).copied().ok_or_else(|| {
                    invalid_input(format!(
                        "node[{index}] input index {} is out of bounds",
                        node.input_index
                    ))
                })?;
                inputs.push(match input {
                    FusedEltwiseInput::Dram { buffer, .. } => FusedEltwiseInput::Dram {
                        buffer,
                        single_tile_broadcast: node.single_tile_broadcast,
                    },
                });
            }
            FusedEltwiseOp::Constant => {}
            _ => {}
        }
    }
    Ok(inputs)
}

fn validate_node_inputs(index: usize, node: &FusedEltwiseNode, expected: usize) -> io::Result<()> {
    if node.input_nodes.len() != expected {
        return Err(invalid_input(format!(
            "node[{index}] {:?} expected {expected} operands, got {}",
            node.op,
            node.input_nodes.len()
        )));
    }
    Ok(())
}

fn validate_node_dtype(
    index: usize,
    node: &FusedEltwiseNode,
    nodes: &[FusedEltwiseNode],
) -> io::Result<()> {
    match node.op {
        FusedEltwiseOp::Input | FusedEltwiseOp::Constant => Ok(()),
        FusedEltwiseOp::Cosine
        | FusedEltwiseOp::Sine
        | FusedEltwiseOp::Negate
        | FusedEltwiseOp::Exponential
        | FusedEltwiseOp::Rsqrt => {
            let input = &nodes[node.input_nodes[0] as usize];
            if input.dtype != node.dtype {
                return Err(invalid_input(format!(
                    "node[{index}] {:?} output dtype must match input dtype, got {:?} -> {:?}",
                    node.op, input.dtype, node.dtype
                )));
            }
            if !is_float_dtype(input.dtype) {
                return Err(invalid_input(format!(
                    "node[{index}] {:?} supports Float16, Float16B, and Float32 inputs, got {:?}",
                    node.op, input.dtype
                )));
            }
            Ok(())
        }
        FusedEltwiseOp::Convert => {
            let input = &nodes[node.input_nodes[0] as usize];
            if !is_convert_dtype(input.dtype) || !is_convert_dtype(node.dtype) {
                return Err(invalid_input(format!(
                    "node[{index}] convert supports Float16B, Float32, Int32, UInt16, and UInt32, got {:?} -> {:?}",
                    input.dtype, node.dtype
                )));
            }
            Ok(())
        }
        FusedEltwiseOp::Add | FusedEltwiseOp::Multiply => {
            validate_binary_input_dtypes(index, node, nodes)?;
            let input_dtype = nodes[node.input_nodes[0] as usize].dtype;
            if node.dtype != input_dtype {
                return Err(invalid_input(format!(
                    "node[{index}] {:?} output dtype must match input dtype, got {:?} -> {:?}",
                    node.op, input_dtype, node.dtype
                )));
            }
            if !matches!(
                input_dtype,
                DType::Float16
                    | DType::Float16B
                    | DType::Float32
                    | DType::Int32
                    | DType::UInt16
                    | DType::UInt32
            ) {
                return Err(invalid_input(format!(
                    "node[{index}] {:?} does not support input dtype {:?}",
                    node.op, input_dtype
                )));
            }
            Ok(())
        }
        FusedEltwiseOp::Subtract => {
            validate_binary_input_dtypes(index, node, nodes)?;
            let input_dtype = nodes[node.input_nodes[0] as usize].dtype;
            if node.dtype != input_dtype {
                return Err(invalid_input(format!(
                    "node[{index}] subtract output dtype must match input dtype, got {:?} -> {:?}",
                    input_dtype, node.dtype
                )));
            }
            if !matches!(
                input_dtype,
                DType::Float16 | DType::Float16B | DType::Float32 | DType::Int32
            ) {
                return Err(invalid_input(format!(
                    "node[{index}] subtract does not support input dtype {:?}",
                    input_dtype
                )));
            }
            Ok(())
        }
        FusedEltwiseOp::Divide | FusedEltwiseOp::Power | FusedEltwiseOp::Max => {
            validate_binary_input_dtypes(index, node, nodes)?;
            let input_dtype = nodes[node.input_nodes[0] as usize].dtype;
            if node.dtype != input_dtype {
                return Err(invalid_input(format!(
                    "node[{index}] {:?} output dtype must match input dtype, got {:?} -> {:?}",
                    node.op, input_dtype, node.dtype
                )));
            }
            if !is_float_dtype(input_dtype) {
                return Err(invalid_input(format!(
                    "node[{index}] {:?} supports Float16, Float16B, and Float32 inputs, got {:?}",
                    node.op, input_dtype
                )));
            }
            Ok(())
        }
        FusedEltwiseOp::Compare(_) => {
            validate_binary_input_dtypes(index, node, nodes)?;
            let input_dtype = nodes[node.input_nodes[0] as usize].dtype;
            if node.dtype != DType::UInt8 {
                return Err(invalid_input(format!(
                    "node[{index}] compare output dtype must be UInt8, got {:?}",
                    node.dtype
                )));
            }
            if !matches!(input_dtype, DType::Float16B | DType::Float32 | DType::Int32) {
                return Err(invalid_input(format!(
                    "node[{index}] compare supports Float16B, Float32, and Int32 inputs, got {:?}",
                    input_dtype
                )));
            }
            Ok(())
        }
    }
}

fn validate_binary_input_dtypes(
    index: usize,
    node: &FusedEltwiseNode,
    nodes: &[FusedEltwiseNode],
) -> io::Result<()> {
    let lhs = nodes[node.input_nodes[0] as usize].dtype;
    let rhs = nodes[node.input_nodes[1] as usize].dtype;
    if lhs != rhs {
        return Err(invalid_input(format!(
            "node[{index}] {:?} input dtypes must match, got {:?} and {:?}",
            node.op, lhs, rhs
        )));
    }
    Ok(())
}

fn fused_eltwise_program(key: FusedEltwiseProgramKey) -> io::Result<Program> {
    let input_count = key.input_dtypes.len();
    let mut reader_dynamic_indices = Vec::with_capacity(input_count);
    reader_dynamic_indices.extend(0..input_count);

    let mut runtime_args = RuntimeArgsBuilder::new(0, vec![0], reader_dynamic_indices, Vec::new());
    for (core_index, &core) in key.cores.iter().enumerate() {
        let (offset, n_tiles) = split_tile_range(key.tile_count, core_index, key.cores.len())?;
        let mut reader_args = vec![0; input_count];
        reader_args.push(offset);
        reader_args.push(n_tiles);
        runtime_args.add_core(core, vec![0, offset, n_tiles], reader_args, vec![n_tiles])?;
    }
    let runtime_args = runtime_args.build()?;

    let (_, intermediate_cbs) = cb_plan(&key.nodes, key.root_node_id)?;
    let mut cbs = Vec::with_capacity(input_count + intermediate_cbs.len() + 1);
    for (index, &dtype) in key.input_dtypes.iter().enumerate() {
        cbs.push(CBConfig::new(index, dtype));
    }
    for (cb, dtype) in intermediate_cbs {
        cbs.push(CBConfig::new(cb as usize, dtype));
    }
    cbs.push(CBConfig::new(16, key.output_dtype));

    let dst_accum_mode = key
        .input_dtypes
        .iter()
        .chain(std::iter::once(&key.output_dtype))
        .chain(key.nodes.iter().map(|node| &node.dtype))
        .any(|dtype| matches!(dtype, DType::Float32 | DType::Int32 | DType::UInt32));

    Ok(Program {
        reader_kernel: reader_source(&key.input_broadcasts, &key.input_dtypes),
        compute_kernel: compute_source(&key)?,
        writer_kernel: WRITER.to_owned(),
        compile: CompileConfig {
            cbs,
            dst_accum_mode,
            ..CompileConfig::default()
        },
        name: format!(
            "fused_eltwise_{}_{}_{}",
            input_count,
            key.nodes.len(),
            key.root_node_id
        ),
        ..Program::new(runtime_args)
    })
}

fn reader_source(input_broadcasts: &[bool], input_dtypes: &[DType]) -> String {
    let input_count = input_broadcasts.len();
    let mut arg_loads = String::new();
    let mut addr_gens = String::new();
    let mut reserves = String::new();
    let mut reads = String::new();
    let mut broadcasts = String::new();
    let mut pushes = String::new();
    for index in 0..input_count {
        arg_loads.push_str(&format!(
            "  uint32_t input_addr_{index} = get_arg_val<uint32_t>({index});\n"
        ));
        addr_gens.push_str(&format!(
            "  constexpr uint32_t cb_input_{index} = tt::CBIndex::c_{index};\n  const InterleavedAddrGenFast<true> input_{index} = {{\n    .bank_base_address = input_addr_{index}, .page_size = get_tile_size(cb_input_{index}), .data_format = get_dataformat(cb_input_{index}),\n  }};\n"
        ));
        reserves.push_str(&format!("    cb_reserve_back(cb_input_{index}, 1);\n"));
        let tile_id = if input_broadcasts[index] {
            "0".to_owned()
        } else {
            "offset + i".to_owned()
        };
        reads.push_str(
            &format!(
                "    noc_async_read_tile(offset + i, input_{index}, get_write_ptr(cb_input_{index}));\n"
            )
            .replace("offset + i", &tile_id),
        );
        if input_broadcasts[index] {
            let mode = match input_dtypes[index] {
                DType::Float16 | DType::Float16B | DType::UInt16 => "true",
                _ => "false",
            };
            broadcasts.push_str(&format!(
                "    replicate_first_element(cb_input_{index}, {mode});\n"
            ));
        }
        pushes.push_str(&format!("    cb_push_back(cb_input_{index}, 1);\n"));
    }

    format!(
        "#include <cstdint>\n\
         \n\
         namespace {{\n\
         void replicate_first_element(uint32_t cb, bool is_16bit) {{\n\
           uint32_t l1_addr = get_write_ptr(cb);\n\
           volatile tt_l1_ptr uint32_t *ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(l1_addr);\n\
           uint32_t packed_value = ptr[0];\n\
           if (is_16bit) {{\n\
             packed_value = (packed_value & 0xffffu) | ((packed_value & 0xffffu) << 16);\n\
           }}\n\
           uint32_t words = get_tile_size(cb) / sizeof(uint32_t);\n\
           for (uint32_t i = 0; i < words; ++i) {{\n\
             ptr[i] = packed_value;\n\
           }}\n\
         }}\n\
         }}  // namespace\n\
         \n\
         void kernel_main() {{\n\
         {arg_loads}\
           uint32_t offset = get_arg_val<uint32_t>({input_count});\n\
           uint32_t n_tiles = get_arg_val<uint32_t>({});\n\
         {addr_gens}\
           for (uint32_t i = 0; i < n_tiles; ++i) {{\n\
         {reserves}\
         {reads}\
             noc_async_read_barrier();\n\
         {broadcasts}\
         {pushes}\
           }}\n\
         }}\n",
        input_count + 1
    )
}

fn compute_source(key: &FusedEltwiseProgramKey) -> io::Result<String> {
    let steps = compute_steps(&key.nodes, key.root_node_id)?;
    Ok(format!(
        "#include <cstdint>\n\
         #include \"compute_kernel_api/common.h\"\n\
         #include \"compute_kernel_api/tile_move_copy.h\"\n\
         #include \"compute_kernel_api/eltwise_unary/eltwise_unary.h\"\n\
         #include \"compute_kernel_api/eltwise_unary/negative.h\"\n\
         #include \"compute_kernel_api/eltwise_unary/exp.h\"\n\
         #include \"compute_kernel_api/eltwise_unary/rsqrt.h\"\n\
         #include \"compute_kernel_api/eltwise_unary/binop_with_scalar.h\"\n\
         #include \"compute_kernel_api/eltwise_unary/rdiv.h\"\n\
         #include \"compute_kernel_api/eltwise_unary/rpow.h\"\n\
         #include \"compute_kernel_api/eltwise_unary/trigonometry.h\"\n\
         #include \"compute_kernel_api/eltwise_unary/typecast.h\"\n\
         #include \"compute_kernel_api/eltwise_binary_sfpu.h\"\n\
         #include \"compute_kernel_api/eltwise_unary/sfpu_split_includes.h\"\n\
         #include \"compute_kernel_api/binary_max_min.h\"\n\
         #include \"compute_kernel_api/add_int_sfpu.h\"\n\
         #include \"compute_kernel_api/sub_int_sfpu.h\"\n\
         #include \"compute_kernel_api/mul_int_sfpu.h\"\n\
         #include \"compute_kernel_api/mul_int32_sfpu.h\"\n\
         #include \"compute_kernel_api/eltwise_unary/comp.h\"\n\
         #include \"compute_kernel_api.h\"\n\
         \n\
         namespace NAMESPACE {{\n\
         ELTWISE_HELPERS\
         void MAIN {{\n\
           uint32_t n_tiles = get_arg_val<uint32_t>(0);\n\
           constexpr uint32_t cb_out = tt::CBIndex::c_16;\n\
         \n\
          unary_op_init_common(tt::CBIndex::c_0, cb_out);\n\
          add_binary_tile_init();\n\
          sub_binary_tile_init();\n\
          mul_binary_tile_init();\n\
          div_binary_tile_init();\n\
          binary_max_tile_init();\n\
          unary_max_tile_init();\n\
          negative_tile_init();\n\
          exp_tile_init();\n\
          rsqrt_tile_init();\n\
          cos_tile_init();\n\
          sin_tile_init();\n\
           binop_with_scalar_tile_init();\n\
           rdiv_tile_init();\n\
           rpow_tile_init();\n\
           power_tile_init();\n\
           power_binary_tile_init();\n\
           FUSED_TYPECAST_INITS\n\
         \n\
           for (uint32_t i = 0; i < n_tiles; ++i) {{\n\
         FUSED_STEPS\
           }}\n\
         }}\n\
         }}  // namespace NAMESPACE\n"
    )
    .replace("ELTWISE_HELPERS", compute_helpers())
    .replace("FUSED_TYPECAST_INITS", &steps.typecast_inits)
    .replace("FUSED_STEPS", &steps.body))
}

fn compute_helpers() -> &'static str {
    r#"
template <DataFormat Format>
ALWI void add_input_init() {
  if constexpr (Format == DataFormat::Float16 || Format == DataFormat::Float16_b ||
                Format == DataFormat::Float32) {
    add_binary_tile_init();
  } else {
    add_int_tile_init();
  }
}

template <DataFormat Format>
ALWI void add_input_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
  if constexpr (Format == DataFormat::Float16 || Format == DataFormat::Float16_b ||
                Format == DataFormat::Float32) {
    add_binary_tile(idst0, idst1, odst);
  } else if constexpr (Format == DataFormat::Int32) {
    add_int32_tile(idst0, idst1, odst);
  } else if constexpr (Format == DataFormat::UInt32) {
    add_uint32_tile(idst0, idst1, odst);
  } else if constexpr (Format == DataFormat::UInt16) {
    add_uint16_tile(idst0, idst1, odst);
  }
}

template <DataFormat Format>
ALWI void subtract_input_init() {
  if constexpr (Format == DataFormat::Float16 || Format == DataFormat::Float16_b ||
                Format == DataFormat::Float32) {
    sub_binary_tile_init();
  } else if constexpr (Format == DataFormat::Int32) {
    sub_int_tile_init();
  }
}

template <DataFormat Format>
ALWI void subtract_input_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
  if constexpr (Format == DataFormat::Float16 || Format == DataFormat::Float16_b ||
                Format == DataFormat::Float32) {
    sub_binary_tile(idst0, idst1, odst);
  } else if constexpr (Format == DataFormat::Int32) {
    sub_int32_tile(idst0, idst1, odst);
  }
}

template <DataFormat Format>
ALWI void multiply_input_init() {
  if constexpr (Format == DataFormat::Float16 || Format == DataFormat::Float16_b ||
                Format == DataFormat::Float32) {
    mul_binary_tile_init();
  } else if constexpr (Format == DataFormat::Int32 || Format == DataFormat::UInt32) {
    mul_int32_tile_init();
  } else if constexpr (Format == DataFormat::UInt16) {
    mul_int_tile_init();
  }
}

template <DataFormat Format>
ALWI void multiply_input_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
  if constexpr (Format == DataFormat::Float16 || Format == DataFormat::Float16_b ||
                Format == DataFormat::Float32) {
    mul_binary_tile(idst0, idst1, odst);
  } else if constexpr (Format == DataFormat::Int32) {
    mul_int32_tile(idst0, idst1, odst);
  } else if constexpr (Format == DataFormat::UInt32) {
    mul_uint32_tile(idst0, idst1, odst);
  } else if constexpr (Format == DataFormat::UInt16) {
    mul_uint16_tile(idst0, idst1, odst);
  }
}

constexpr DataFormat binary_input_data_format(uint32_t cb_lhs, uint32_t cb_out) {
#ifdef UCK_CHLKC_PACK
  return static_cast<DataFormat>((uint)pack_src_format[cb_out]);
#else
  return static_cast<DataFormat>((uint)unpack_src_format[cb_lhs]);
#endif
}

enum class CompareDirection : uint32_t {
  Eq,
  Ne,
  Ge,
  Gt,
  Le,
  Lt,
};

template <bool Int32Input>
ALWI void compare_sub_init() {
  if constexpr (Int32Input) {
    sub_int_tile_init();
  } else {
    sub_binary_tile_init();
  }
}

template <bool Int32Input>
ALWI void compare_sub_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
  if constexpr (Int32Input) {
    sub_int32_tile(idst0, idst1, odst);
  } else {
    sub_binary_tile(idst0, idst1, odst);
  }
}

ALWI void compare_zero_init(CompareDirection direction) {
  switch (direction) {
    case CompareDirection::Eq: eqz_tile_init(); break;
    case CompareDirection::Ne: nez_tile_init(); break;
    case CompareDirection::Ge: gez_tile_init(); break;
    case CompareDirection::Gt: gtz_tile_init(); break;
    case CompareDirection::Le: lez_tile_init(); break;
    case CompareDirection::Lt: ltz_tile_init(); break;
    default: break;
  }
}

template <bool Int32Input>
ALWI void compare_zero_tile(CompareDirection direction, uint32_t idst) {
  switch (direction) {
    case CompareDirection::Eq:
      if constexpr (Int32Input) {
        eqz_tile_int32(idst);
      } else {
        eqz_tile(idst);
      }
      break;
    case CompareDirection::Ne:
      if constexpr (Int32Input) {
        nez_tile_int32(idst);
      } else {
        nez_tile(idst);
      }
      break;
    case CompareDirection::Ge:
      if constexpr (Int32Input) {
        gez_tile_int32(idst);
      } else {
        gez_tile(idst);
      }
      break;
    case CompareDirection::Gt:
      if constexpr (Int32Input) {
        gtz_tile_int32(idst);
      } else {
        gtz_tile(idst);
      }
      break;
    case CompareDirection::Le:
      if constexpr (Int32Input) {
        lez_tile_int32(idst);
      } else {
        lez_tile(idst);
      }
      break;
    case CompareDirection::Lt:
      if constexpr (Int32Input) {
        ltz_tile_int32(idst);
      } else {
        ltz_tile(idst);
      }
      break;
    default: break;
  }
}

"#
}

struct ComputeSteps {
    body: String,
    typecast_inits: String,
}

fn compute_steps(nodes: &[FusedEltwiseNode], root_node_id: u32) -> io::Result<ComputeSteps> {
    let mut remaining_uses = vec![0u32; nodes.len()];
    for node in nodes {
        for &input_node in &node.input_nodes {
            let index = usize::try_from(input_node)
                .map_err(|_| invalid_input(format!("node id out of range: {input_node}")))?;
            if index >= nodes.len() {
                return Err(invalid_input(format!(
                    "node id out of bounds: {input_node}"
                )));
            }
            remaining_uses[index] += 1;
        }
    }

    let (node_cbs, _) = cb_plan(nodes, root_node_id)?;
    let mut body = String::new();
    let mut typecast_inits = String::new();

    for (index, node) in nodes.iter().enumerate() {
        match node.op {
            FusedEltwiseOp::Input | FusedEltwiseOp::Constant => {}
            FusedEltwiseOp::Negate
            | FusedEltwiseOp::Cosine
            | FusedEltwiseOp::Sine
            | FusedEltwiseOp::Exponential
            | FusedEltwiseOp::Rsqrt
            | FusedEltwiseOp::Convert => {
                let input = node.input_nodes[0] as usize;
                let input_cb = cb_for_node(&node_cbs, input)?;
                let output_cb = cb_for_node(&node_cbs, index)?;
                append_waits(&mut body, &[input_cb]);
                let init = match node.op {
                    FusedEltwiseOp::Negate => "negative_tile_init();",
                    FusedEltwiseOp::Cosine => "cos_tile_init();",
                    FusedEltwiseOp::Sine => "sin_tile_init();",
                    FusedEltwiseOp::Exponential => "exp_tile_init();",
                    FusedEltwiseOp::Rsqrt => "rsqrt_tile_init();",
                    FusedEltwiseOp::Convert => "",
                    _ => unreachable!(),
                };
                body.push_str(&format!(
                    "    {init}\n    cb_reserve_back(tt::CBIndex::c_{output_cb}, 1);\n    tile_regs_acquire();\n    copy_tile_to_dst_init_short(tt::CBIndex::c_{input_cb});\n    copy_tile(tt::CBIndex::c_{input_cb}, 0, 0);\n"
                ));
                match node.op {
                    FusedEltwiseOp::Negate => {
                        body.push_str("    negative_tile(0);\n");
                    }
                    FusedEltwiseOp::Cosine => {
                        body.push_str("    cos_tile(0);\n");
                    }
                    FusedEltwiseOp::Sine => {
                        body.push_str("    sin_tile(0);\n");
                    }
                    FusedEltwiseOp::Exponential => {
                        body.push_str("    exp_tile(0);\n");
                    }
                    FusedEltwiseOp::Rsqrt => {
                        body.push_str("    rsqrt_tile(0);\n");
                    }
                    FusedEltwiseOp::Convert => {
                        let from = nodes[input].dtype as u32;
                        let to = node.dtype as u32;
                        typecast_inits
                            .push_str(&format!("typecast_tile_init<{from}, {to}>();\n           "));
                        body.push_str(&format!("    typecast_tile<{from}, {to}>(0);\n"));
                    }
                    _ => unreachable!(),
                }
                append_pack_and_pop(
                    &mut body,
                    output_cb,
                    &[input],
                    &node_cbs,
                    &mut remaining_uses,
                )?;
            }
            FusedEltwiseOp::Add
            | FusedEltwiseOp::Subtract
            | FusedEltwiseOp::Multiply
            | FusedEltwiseOp::Divide
            | FusedEltwiseOp::Power
            | FusedEltwiseOp::Max
            | FusedEltwiseOp::Compare(_) => {
                let lhs = node.input_nodes[0] as usize;
                let rhs = node.input_nodes[1] as usize;
                if let FusedEltwiseOp::Compare(direction) = node.op {
                    if let Some((value_node, scalar, scalar_direction)) =
                        scalar_compare_op(nodes, lhs, rhs, direction)
                    {
                        let value_cb = cb_for_node(&node_cbs, value_node)?;
                        let output_cb = cb_for_node(&node_cbs, index)?;
                        let int32_input = nodes[value_node].dtype == DType::Int32;
                        let init = unary_compare_init(scalar_direction);
                        let call = unary_compare_call(scalar_direction, int32_input);
                        append_waits(&mut body, &[value_cb]);
                        body.push_str(&format!(
                            "    {init}();\n    cb_reserve_back(tt::CBIndex::c_{output_cb}, 1);\n    tile_regs_acquire();\n    copy_tile_to_dst_init_short(tt::CBIndex::c_{value_cb});\n    copy_tile(tt::CBIndex::c_{value_cb}, 0, 0);\n    {call}(0, {scalar});\n"
                        ));
                        append_pack_and_pop(
                            &mut body,
                            output_cb,
                            &[value_node],
                            &node_cbs,
                            &mut remaining_uses,
                        )?;
                        continue;
                    }
                }
                if let Some((value_node, scalar, scalar_op)) =
                    scalar_binary_op(nodes, node.op, lhs, rhs)
                {
                    let value_cb = cb_for_node(&node_cbs, value_node)?;
                    let output_cb = cb_for_node(&node_cbs, index)?;
                    let scalar_init = scalar_op_init(scalar_op);
                    append_waits(&mut body, &[value_cb]);
                    body.push_str(&format!(
                        "    {scalar_init}();\n    cb_reserve_back(tt::CBIndex::c_{output_cb}, 1);\n    tile_regs_acquire();\n    copy_tile_to_dst_init_short(tt::CBIndex::c_{value_cb});\n    copy_tile(tt::CBIndex::c_{value_cb}, 0, 0);\n    {scalar_op}(0, {scalar});\n"
                    ));
                    append_pack_and_pop(
                        &mut body,
                        output_cb,
                        &[value_node],
                        &node_cbs,
                        &mut remaining_uses,
                    )?;
                    continue;
                }
                let lhs_cb = cb_for_node(&node_cbs, lhs)?;
                let rhs_cb = cb_for_node(&node_cbs, rhs)?;
                let output_cb = cb_for_node(&node_cbs, index)?;
                if let FusedEltwiseOp::Compare(direction) = node.op {
                    let int32_input = bool_literal(nodes[lhs].dtype == DType::Int32);
                    let direction = compare_direction_variant(direction);
                    append_waits(&mut body, &[lhs_cb, rhs_cb]);
                    body.push_str(&format!(
                        "    compare_sub_init<{int32_input}>();\n    compare_zero_init(CompareDirection::{direction});\n    cb_reserve_back(tt::CBIndex::c_{output_cb}, 1);\n    tile_regs_acquire();\n    copy_tile_to_dst_init_short_with_dt(tt::CBIndex::c_{rhs_cb}, tt::CBIndex::c_{lhs_cb});\n    copy_tile(tt::CBIndex::c_{lhs_cb}, 0, 0);\n    copy_tile_to_dst_init_short_with_dt(tt::CBIndex::c_{lhs_cb}, tt::CBIndex::c_{rhs_cb});\n    copy_tile(tt::CBIndex::c_{rhs_cb}, 0, 1);\n    compare_sub_tile<{int32_input}>(0, 1, 0);\n    compare_zero_tile<{int32_input}>(CompareDirection::{direction}, 0);\n"
                    ));
                    append_pack_and_pop(
                        &mut body,
                        output_cb,
                        &[lhs, rhs],
                        &node_cbs,
                        &mut remaining_uses,
                    )?;
                    continue;
                }
                if matches!(
                    node.op,
                    FusedEltwiseOp::Add | FusedEltwiseOp::Subtract | FusedEltwiseOp::Multiply
                ) {
                    let helper = match node.op {
                        FusedEltwiseOp::Add => "add_input",
                        FusedEltwiseOp::Subtract => "subtract_input",
                        FusedEltwiseOp::Multiply => "multiply_input",
                        _ => unreachable!(),
                    };
                    append_waits(&mut body, &[lhs_cb, rhs_cb]);
                    body.push_str(&format!(
                        "    constexpr DataFormat input_format_{index} = binary_input_data_format(tt::CBIndex::c_{lhs_cb}, tt::CBIndex::c_{output_cb});\n    {helper}_init<input_format_{index}>();\n    cb_reserve_back(tt::CBIndex::c_{output_cb}, 1);\n    tile_regs_acquire();\n    copy_tile_to_dst_init_short_with_dt(tt::CBIndex::c_{rhs_cb}, tt::CBIndex::c_{lhs_cb});\n    copy_tile(tt::CBIndex::c_{lhs_cb}, 0, 0);\n    copy_tile_to_dst_init_short_with_dt(tt::CBIndex::c_{lhs_cb}, tt::CBIndex::c_{rhs_cb});\n    copy_tile(tt::CBIndex::c_{rhs_cb}, 0, 1);\n    {helper}_tile<input_format_{index}>(0, 1, 0);\n"
                    ));
                    append_pack_and_pop(
                        &mut body,
                        output_cb,
                        &[lhs, rhs],
                        &node_cbs,
                        &mut remaining_uses,
                    )?;
                    continue;
                }
                let call = match node.op {
                    FusedEltwiseOp::Divide => "div_binary_tile",
                    FusedEltwiseOp::Power => "power_binary_tile",
                    FusedEltwiseOp::Max => "binary_max_tile",
                    _ => unreachable!(),
                };
                let init = match node.op {
                    FusedEltwiseOp::Divide => "div_binary_tile_init",
                    FusedEltwiseOp::Power => "power_binary_tile_init",
                    FusedEltwiseOp::Max => "binary_max_tile_init",
                    _ => unreachable!(),
                };
                append_waits(&mut body, &[lhs_cb, rhs_cb]);
                body.push_str(&format!(
                    "    {init}();\n    cb_reserve_back(tt::CBIndex::c_{output_cb}, 1);\n    tile_regs_acquire();\n    copy_tile_to_dst_init_short_with_dt(tt::CBIndex::c_{rhs_cb}, tt::CBIndex::c_{lhs_cb});\n    copy_tile(tt::CBIndex::c_{lhs_cb}, 0, 0);\n    copy_tile_to_dst_init_short_with_dt(tt::CBIndex::c_{lhs_cb}, tt::CBIndex::c_{rhs_cb});\n    copy_tile(tt::CBIndex::c_{rhs_cb}, 0, 1);\n    {call}(0, 1, 0);\n"
                ));
                append_pack_and_pop(
                    &mut body,
                    output_cb,
                    &[lhs, rhs],
                    &node_cbs,
                    &mut remaining_uses,
                )?;
            }
        }
    }

    Ok(ComputeSteps {
        body,
        typecast_inits,
    })
}

fn cb_plan(
    nodes: &[FusedEltwiseNode],
    root_node_id: u32,
) -> io::Result<(Vec<Option<u32>>, Vec<(u32, DType)>)> {
    let mut node_cbs = vec![None; nodes.len()];
    let mut leaf_count = 0u32;
    for (index, node) in nodes.iter().enumerate() {
        if node.op == FusedEltwiseOp::Input {
            if leaf_count >= 16 {
                return Err(invalid_input("fused eltwise needs too many input CBs"));
            }
            node_cbs[index] = Some(leaf_count);
            leaf_count += 1;
        }
    }

    let root_index = usize::try_from(root_node_id)
        .map_err(|_| invalid_input(format!("root node id is out of range: {root_node_id}")))?;
    let mut next_cb = leaf_count;
    let mut intermediate_cbs = Vec::new();
    for (index, node) in nodes.iter().enumerate() {
        if matches!(node.op, FusedEltwiseOp::Input | FusedEltwiseOp::Constant) {
            continue;
        }
        if index == root_index {
            node_cbs[index] = Some(16);
        } else {
            if next_cb >= 16 {
                return Err(invalid_input(
                    "fused eltwise needs too many intermediate CBs",
                ));
            }
            node_cbs[index] = Some(next_cb);
            intermediate_cbs.push((next_cb, node.dtype));
            next_cb += 1;
        }
    }
    Ok((node_cbs, intermediate_cbs))
}

fn cb_for_node(node_cbs: &[Option<u32>], node: usize) -> io::Result<u32> {
    node_cbs
        .get(node)
        .and_then(|cb| *cb)
        .ok_or_else(|| invalid_input(format!("node {node} does not have a CB")))
}

fn append_waits(body: &mut String, cbs: &[u32]) {
    let mut waited = Vec::new();
    for &cb in cbs {
        if waited.contains(&cb) {
            continue;
        }
        waited.push(cb);
        body.push_str(&format!("    cb_wait_front(tt::CBIndex::c_{cb}, 1);\n"));
    }
}

fn append_pack_and_pop(
    body: &mut String,
    output_cb: u32,
    input_nodes: &[usize],
    node_cbs: &[Option<u32>],
    remaining_uses: &mut [u32],
) -> io::Result<()> {
    body.push_str(&format!(
        "    tile_regs_commit();\n    tile_regs_wait();\n    pack_tile(0, tt::CBIndex::c_{output_cb});\n    tile_regs_release();\n"
    ));

    let mut consumed = Vec::<(usize, u32)>::new();
    for &node in input_nodes {
        if let Some((_, count)) = consumed.iter_mut().find(|(existing, _)| *existing == node) {
            *count += 1;
        } else {
            consumed.push((node, 1));
        }
    }
    for (node, count) in consumed {
        remaining_uses[node] = remaining_uses[node]
            .checked_sub(count)
            .ok_or_else(|| invalid_input(format!("node {node} use count underflow")))?;
        if remaining_uses[node] == 0 {
            if let Some(cb) = node_cbs[node] {
                body.push_str(&format!("    cb_pop_front(tt::CBIndex::c_{cb}, 1);\n"));
            }
        }
    }
    body.push_str(&format!(
        "    cb_push_back(tt::CBIndex::c_{output_cb}, 1);\n"
    ));
    Ok(())
}

fn scalar_binary_op(
    nodes: &[FusedEltwiseNode],
    op: FusedEltwiseOp,
    lhs: usize,
    rhs: usize,
) -> Option<(usize, u32, &'static str)> {
    let lhs_constant = constant_scalar_bits(nodes, lhs);
    let rhs_constant = constant_scalar_bits(nodes, rhs);
    let scalar_op = |value_node: usize,
                     scalar: u32,
                     float_op: &'static str,
                     int32_op: &'static str|
     -> Option<(usize, u32, &'static str)> {
        let op = match nodes[value_node].dtype {
            DType::Float16 | DType::Float16B | DType::Float32 => float_op,
            DType::Int32 => int32_op,
            _ => return None,
        };
        Some((value_node, scalar, op))
    };
    match (op, lhs_constant, rhs_constant) {
        (FusedEltwiseOp::Add, None, Some(scalar)) => {
            scalar_op(lhs, scalar, "add_unary_tile", "add_unary_tile_int32")
        }
        (FusedEltwiseOp::Add, Some(scalar), None) => {
            scalar_op(rhs, scalar, "add_unary_tile", "add_unary_tile_int32")
        }
        (FusedEltwiseOp::Subtract, None, Some(scalar)) => {
            scalar_op(lhs, scalar, "sub_unary_tile", "sub_unary_tile_int32")
        }
        (FusedEltwiseOp::Subtract, Some(scalar), None)
            if matches!(
                nodes[rhs].dtype,
                DType::Float16 | DType::Float16B | DType::Float32
            ) =>
        {
            Some((rhs, scalar, "rsub_unary_tile"))
        }
        (FusedEltwiseOp::Multiply, None, Some(scalar))
            if matches!(
                nodes[lhs].dtype,
                DType::Float16 | DType::Float16B | DType::Float32
            ) =>
        {
            Some((lhs, scalar, "mul_unary_tile"))
        }
        (FusedEltwiseOp::Multiply, Some(scalar), None)
            if matches!(
                nodes[rhs].dtype,
                DType::Float16 | DType::Float16B | DType::Float32
            ) =>
        {
            Some((rhs, scalar, "mul_unary_tile"))
        }
        (FusedEltwiseOp::Divide, None, Some(scalar))
            if matches!(
                nodes[lhs].dtype,
                DType::Float16 | DType::Float16B | DType::Float32
            ) =>
        {
            Some((
                lhs,
                (1.0f32 / f32::from_bits(scalar)).to_bits(),
                "div_unary_tile",
            ))
        }
        (FusedEltwiseOp::Divide, Some(scalar), None)
            if matches!(
                nodes[rhs].dtype,
                DType::Float16 | DType::Float16B | DType::Float32
            ) =>
        {
            Some((rhs, scalar, "rdiv_tile"))
        }
        (FusedEltwiseOp::Power, None, Some(scalar))
            if matches!(
                nodes[lhs].dtype,
                DType::Float16 | DType::Float16B | DType::Float32
            ) =>
        {
            Some((lhs, scalar, "power_tile"))
        }
        (FusedEltwiseOp::Power, Some(scalar), None)
            if matches!(
                nodes[rhs].dtype,
                DType::Float16 | DType::Float16B | DType::Float32
            ) =>
        {
            Some((rhs, scalar, "rpow_tile"))
        }
        (FusedEltwiseOp::Max, None, Some(scalar))
            if matches!(
                nodes[lhs].dtype,
                DType::Float16 | DType::Float16B | DType::Float32
            ) =>
        {
            Some((lhs, scalar, "unary_max_tile"))
        }
        (FusedEltwiseOp::Max, Some(scalar), None)
            if matches!(
                nodes[rhs].dtype,
                DType::Float16 | DType::Float16B | DType::Float32
            ) =>
        {
            Some((rhs, scalar, "unary_max_tile"))
        }
        _ => None,
    }
}

fn scalar_compare_op(
    nodes: &[FusedEltwiseNode],
    lhs: usize,
    rhs: usize,
    direction: CompareDirection,
) -> Option<(usize, u32, CompareDirection)> {
    let lhs_constant = constant_scalar_bits(nodes, lhs);
    let rhs_constant = constant_scalar_bits(nodes, rhs);
    match (lhs_constant, rhs_constant) {
        (None, Some(scalar)) => Some((lhs, scalar, direction)),
        (Some(scalar), None) => Some((rhs, scalar, reverse_compare_direction(direction))),
        _ => None,
    }
}

fn reverse_compare_direction(direction: CompareDirection) -> CompareDirection {
    match direction {
        CompareDirection::Eq => CompareDirection::Eq,
        CompareDirection::Ne => CompareDirection::Ne,
        CompareDirection::Ge => CompareDirection::Le,
        CompareDirection::Gt => CompareDirection::Lt,
        CompareDirection::Le => CompareDirection::Ge,
        CompareDirection::Lt => CompareDirection::Gt,
    }
}

fn scalar_op_init(scalar_op: &str) -> &'static str {
    match scalar_op {
        "rdiv_tile" => "rdiv_tile_init",
        "rpow_tile" => "rpow_tile_init",
        "power_tile" => "power_tile_init",
        _ => "binop_with_scalar_tile_init",
    }
}

fn unary_compare_init(direction: CompareDirection) -> &'static str {
    match direction {
        CompareDirection::Eq => "unary_eq_tile_init",
        CompareDirection::Ne => "unary_ne_tile_init",
        CompareDirection::Ge => "unary_ge_tile_init",
        CompareDirection::Gt => "unary_gt_tile_init",
        CompareDirection::Le => "unary_le_tile_init",
        CompareDirection::Lt => "unary_lt_tile_init",
    }
}

fn unary_compare_call(direction: CompareDirection, int32_input: bool) -> &'static str {
    match (direction, int32_input) {
        (CompareDirection::Eq, false) => "unary_eq_tile",
        (CompareDirection::Ne, false) => "unary_ne_tile",
        (CompareDirection::Ge, false) => "unary_ge_tile",
        (CompareDirection::Gt, false) => "unary_gt_tile",
        (CompareDirection::Le, false) => "unary_le_tile",
        (CompareDirection::Lt, false) => "unary_lt_tile",
        (CompareDirection::Eq, true) => "unary_eq_tile_int32",
        (CompareDirection::Ne, true) => "unary_ne_tile_int32",
        (CompareDirection::Ge, true) => "unary_ge_tile_int32",
        (CompareDirection::Gt, true) => "unary_gt_tile_int32",
        (CompareDirection::Le, true) => "unary_le_tile_int32",
        (CompareDirection::Lt, true) => "unary_lt_tile_int32",
    }
}

fn constant_scalar_bits(nodes: &[FusedEltwiseNode], index: usize) -> Option<u32> {
    let node = &nodes[index];
    (node.op == FusedEltwiseOp::Constant).then(|| match node.dtype {
        DType::Float32 => node.packed_value,
        DType::Float16B => (node.packed_value & 0xffff) << 16,
        DType::Float16 => f16_to_f32_bits((node.packed_value & 0xffff) as u16),
        _ => node.packed_value,
    })
}

fn bool_literal(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

fn compare_direction_variant(direction: CompareDirection) -> &'static str {
    match direction {
        CompareDirection::Eq => "Eq",
        CompareDirection::Ne => "Ne",
        CompareDirection::Ge => "Ge",
        CompareDirection::Gt => "Gt",
        CompareDirection::Le => "Le",
        CompareDirection::Lt => "Lt",
    }
}

fn f16_to_f32_bits(value: u16) -> u32 {
    let sign = ((value & 0x8000) as u32) << 16;
    let exponent = ((value >> 10) & 0x1f) as i32;
    let fraction = (value & 0x03ff) as u32;
    match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let mut fraction = fraction;
            let mut exponent = -14;
            while (fraction & 0x0400) == 0 {
                fraction <<= 1;
                exponent -= 1;
            }
            fraction &= 0x03ff;
            sign | (((exponent + 127) as u32) << 23) | (fraction << 13)
        }
        31 => sign | (0xff << 23) | (fraction << 13),
        _ => sign | (((exponent - 15 + 127) as u32) << 23) | (fraction << 13),
    }
}

fn is_supported_leaf_dtype(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::Float16
            | DType::Float16B
            | DType::Float32
            | DType::Int32
            | DType::UInt16
            | DType::UInt32
    )
}

fn is_float_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::Float16 | DType::Float16B | DType::Float32)
}

fn is_convert_dtype(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::Float16B | DType::Float32 | DType::Int32 | DType::UInt16 | DType::UInt32
    )
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn u32_arg<T>(value: T, name: &str) -> io::Result<u32>
where
    T: TryInto<u32> + Copy + Display,
{
    value
        .try_into()
        .map_err(|_| invalid_input(format!("{name} does not fit in u32: {value}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(op: FusedEltwiseOp, input_nodes: Vec<u32>) -> FusedEltwiseNode {
        FusedEltwiseNode {
            op,
            input_nodes,
            input_index: 0,
            packed_value: 0,
            dtype: DType::Float16B,
            single_tile_broadcast: false,
        }
    }

    #[test]
    fn compute_steps_handles_constant_left_divide_as_rdiv() {
        let mut constant = node(FusedEltwiseOp::Constant, Vec::new());
        constant.packed_value = 0x3f80_3f80;

        let nodes = vec![
            node(FusedEltwiseOp::Input, Vec::new()),
            constant,
            node(FusedEltwiseOp::Divide, vec![1, 0]),
        ];
        let steps = compute_steps(&nodes, 2).expect("constant / value should lower");

        assert!(steps.body.contains("rdiv_tile_init();"));
        assert!(steps.body.contains("rdiv_tile(0, 1065353216);"));
    }

    #[test]
    fn compute_steps_handles_constant_right_power_as_unary_power() {
        let mut constant = node(FusedEltwiseOp::Constant, Vec::new());
        constant.packed_value = 0x4000_4000;

        let nodes = vec![
            node(FusedEltwiseOp::Input, Vec::new()),
            constant,
            node(FusedEltwiseOp::Power, vec![0, 1]),
        ];
        let steps = compute_steps(&nodes, 2).expect("value ** constant should lower");

        assert!(steps.body.contains("power_tile_init();"));
        assert!(steps.body.contains("power_tile(0, 1073741824);"));
    }

    #[test]
    fn compute_steps_handles_constant_right_compare_as_unary_compare() {
        let mut constant = node(FusedEltwiseOp::Constant, Vec::new());
        constant.packed_value = 0;

        let nodes = vec![
            node(FusedEltwiseOp::Input, Vec::new()),
            constant,
            node(FusedEltwiseOp::Compare(CompareDirection::Gt), vec![0, 1]),
        ];
        let steps = compute_steps(&nodes, 2).expect("value > constant should lower");

        assert!(steps.body.contains("unary_gt_tile_init();"));
        assert!(steps.body.contains("unary_gt_tile(0, 0);"));
    }

    #[test]
    fn compute_steps_reverses_constant_left_compare() {
        let mut constant = node(FusedEltwiseOp::Constant, Vec::new());
        constant.packed_value = 0;

        let nodes = vec![
            constant,
            node(FusedEltwiseOp::Input, Vec::new()),
            node(FusedEltwiseOp::Compare(CompareDirection::Gt), vec![0, 1]),
        ];
        let steps = compute_steps(&nodes, 2).expect("constant > value should lower");

        assert!(steps.body.contains("unary_lt_tile_init();"));
        assert!(steps.body.contains("unary_lt_tile(0, 0);"));
    }
}
