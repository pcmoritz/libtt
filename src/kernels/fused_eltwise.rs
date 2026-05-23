use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, split_tile_range, Kernel, RuntimeArgsBuilder};
use std::fmt::Display;
use std::io;

const WRITER: &str = include_str!("../../kernels/binary_eltwise_writer.cc");
const MAX_FUSED_INPUTS: usize = 8;
const MAX_FUSED_NODES: usize = 8;
const MAX_DST_SLOTS: u32 = 8;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) enum FusedEltwiseOp {
    Input,
    Constant,
    Add,
    Subtract,
    Multiply,
    Divide,
    Max,
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
    for (index, input) in external_inputs.iter().enumerate() {
        let FusedEltwiseInput::Dram { buffer, .. } = input;
        if !is_supported_float(buffer.dtype) {
            return Err(invalid_input(format!(
                "input[{index}] dtype {:?} is not supported by fused eltwise",
                buffer.dtype
            )));
        }
    }

    for (index, node) in nodes.iter().enumerate() {
        if !is_supported_float(node.dtype) {
            return Err(invalid_input(format!(
                "node[{index}] dtype {:?} is not supported by fused eltwise",
                node.dtype
            )));
        }
        match node.op {
            FusedEltwiseOp::Input => {
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
                if !node.input_nodes.is_empty() {
                    return Err(invalid_input(format!(
                        "node[{index}] constant node must not have operands"
                    )));
                }
            }
            FusedEltwiseOp::Negate
            | FusedEltwiseOp::Exponential
            | FusedEltwiseOp::Rsqrt
            | FusedEltwiseOp::Convert => validate_node_inputs(index, node, 1)?,
            FusedEltwiseOp::Add
            | FusedEltwiseOp::Subtract
            | FusedEltwiseOp::Multiply
            | FusedEltwiseOp::Divide
            | FusedEltwiseOp::Max => validate_node_inputs(index, node, 2)?,
        }
        for &input_node in &node.input_nodes {
            if usize::try_from(input_node).map_or(true, |input| input >= index) {
                return Err(invalid_input(format!(
                    "node[{index}] references non-prior input node {input_node}"
                )));
            }
        }
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

    let mut cbs = Vec::with_capacity(input_count + 1);
    for (index, &dtype) in key.input_dtypes.iter().enumerate() {
        cbs.push(CBConfig::new(index, dtype));
    }
    cbs.push(CBConfig::new(16, key.output_dtype));

    let dst_accum_mode = key
        .input_dtypes
        .iter()
        .chain(std::iter::once(&key.output_dtype))
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
         #include \"compute_kernel_api/eltwise_unary/typecast.h\"\n\
         #include \"compute_kernel_api/eltwise_binary_sfpu.h\"\n\
         #include \"compute_kernel_api/eltwise_unary/sfpu_split_includes.h\"\n\
         #include \"compute_kernel_api/binary_max_min.h\"\n\
         #include \"compute_kernel_api.h\"\n\
         \n\
         namespace NAMESPACE {{\n\
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
           binop_with_scalar_tile_init();\n\
           FUSED_TYPECAST_INITS\n\
         \n\
           for (uint32_t i = 0; i < n_tiles; ++i) {{\n\
         FUSED_WAITS\
             cb_reserve_back(cb_out, 1);\n\
         \n\
             tile_regs_acquire();\n\
         FUSED_STEPS\
             tile_regs_commit();\n\
         \n\
             tile_regs_wait();\n\
             pack_tile(FUSED_ROOT_SLOT, cb_out);\n\
             tile_regs_release();\n\
         \n\
         FUSED_POPS\
             cb_push_back(cb_out, 1);\n\
           }}\n\
         }}\n\
         }}  // namespace NAMESPACE\n"
    )
    .replace("FUSED_TYPECAST_INITS", &steps.typecast_inits)
    .replace("FUSED_WAITS", &steps.waits)
    .replace("FUSED_STEPS", &steps.body)
    .replace("FUSED_ROOT_SLOT", &steps.root_slot.to_string())
    .replace("FUSED_POPS", &steps.pops))
}

struct ComputeSteps {
    waits: String,
    pops: String,
    body: String,
    typecast_inits: String,
    root_slot: u32,
}

fn compute_steps(nodes: &[FusedEltwiseNode], root_node_id: u32) -> io::Result<ComputeSteps> {
    let mut use_counts = vec![0u32; nodes.len()];
    for node in nodes {
        for &input_node in &node.input_nodes {
            let index = usize::try_from(input_node)
                .map_err(|_| invalid_input(format!("node id out of range: {input_node}")))?;
            if index >= nodes.len() {
                return Err(invalid_input(format!(
                    "node id out of bounds: {input_node}"
                )));
            }
            use_counts[index] += 1;
        }
    }

    let mut slots = vec![None; nodes.len()];
    let mut leaf_count = 0u32;
    let mut waits = String::new();
    let mut pops = String::new();
    let mut body = String::new();
    let mut typecast_inits = String::new();
    let mut next_slot = 0u32;

    for (index, node) in nodes.iter().enumerate() {
        match node.op {
            FusedEltwiseOp::Input => {
                let cb = leaf_count;
                leaf_count += 1;
                next_slot = next_slot.max(cb + 1);
                if leaf_count > MAX_DST_SLOTS {
                    return Err(invalid_input(format!(
                        "fused eltwise needs {leaf_count} leaf dst slots, max is {MAX_DST_SLOTS}"
                    )));
                }
                slots[index] = Some(cb);
                waits.push_str(&format!("    cb_wait_front(tt::CBIndex::c_{cb}, 1);\n"));
                pops.push_str(&format!("    cb_pop_front(tt::CBIndex::c_{cb}, 1);\n"));
                body.push_str(&format!(
                    "    copy_tile_to_dst_init_short(tt::CBIndex::c_{cb});\n    copy_tile(tt::CBIndex::c_{cb}, 0, {cb});\n"
                ));
            }
            FusedEltwiseOp::Constant => {}
            FusedEltwiseOp::Negate
            | FusedEltwiseOp::Exponential
            | FusedEltwiseOp::Rsqrt
            | FusedEltwiseOp::Convert => {
                let input = node.input_nodes[0] as usize;
                if use_counts[input] != 1 {
                    return Err(invalid_input(format!(
                        "fused eltwise unary node {index} cannot overwrite shared input node {input}"
                    )));
                }
                let slot = slots[input].ok_or_else(|| {
                    invalid_input(format!("input slot for node {input} is not available"))
                })?;
                slots[index] = Some(slot);
                match node.op {
                    FusedEltwiseOp::Negate => {
                        body.push_str(&format!("    negative_tile({slot});\n"));
                    }
                    FusedEltwiseOp::Exponential => {
                        body.push_str(&format!("    exp_tile({slot});\n"));
                    }
                    FusedEltwiseOp::Rsqrt => {
                        body.push_str(&format!("    rsqrt_tile({slot});\n"));
                    }
                    FusedEltwiseOp::Convert => {
                        let from = nodes[input].dtype as u32;
                        let to = node.dtype as u32;
                        typecast_inits
                            .push_str(&format!("typecast_tile_init<{from}, {to}>();\n           "));
                        body.push_str(&format!("    typecast_tile<{from}, {to}>({slot});\n"));
                    }
                    _ => unreachable!(),
                }
            }
            FusedEltwiseOp::Add
            | FusedEltwiseOp::Subtract
            | FusedEltwiseOp::Multiply
            | FusedEltwiseOp::Divide
            | FusedEltwiseOp::Max => {
                let lhs = node.input_nodes[0] as usize;
                let rhs = node.input_nodes[1] as usize;
                if let Some((value_node, scalar, scalar_op)) =
                    scalar_binary_op(nodes, node.op, lhs, rhs)
                {
                    if use_counts[value_node] != 1 {
                        return Err(invalid_input(format!(
                            "fused eltwise scalar op node {index} cannot overwrite shared input node {value_node}"
                        )));
                    }
                    let slot = slots[value_node].ok_or_else(|| {
                        invalid_input(format!(
                            "scalar input slot for node {value_node} is not available"
                        ))
                    })?;
                    slots[index] = Some(slot);
                    body.push_str(&format!("    {scalar_op}({slot}, {scalar});\n"));
                    continue;
                }
                let lhs_slot = slots[lhs].ok_or_else(|| {
                    invalid_input(format!("lhs slot for node {lhs} is not available"))
                })?;
                let rhs_slot = slots[rhs].ok_or_else(|| {
                    invalid_input(format!("rhs slot for node {rhs} is not available"))
                })?;
                let out_slot = if use_counts[lhs] == 1 {
                    lhs_slot
                } else if use_counts[rhs] == 1 {
                    rhs_slot
                } else {
                    let slot = next_slot;
                    next_slot += 1;
                    if next_slot > MAX_DST_SLOTS {
                        return Err(invalid_input(format!(
                            "fused eltwise needs more than {MAX_DST_SLOTS} dst slots"
                        )));
                    }
                    slot
                };
                slots[index] = Some(out_slot);
                let call = match node.op {
                    FusedEltwiseOp::Add => "add_binary_tile",
                    FusedEltwiseOp::Subtract => "sub_binary_tile",
                    FusedEltwiseOp::Multiply => "mul_binary_tile",
                    FusedEltwiseOp::Divide => "div_binary_tile",
                    FusedEltwiseOp::Max => "binary_max_tile",
                    _ => unreachable!(),
                };
                body.push_str(&format!(
                    "    {call}({lhs_slot}, {rhs_slot}, {out_slot});\n"
                ));
            }
        }
    }

    let root_index = root_node_id as usize;
    let root_slot = slots
        .get(root_index)
        .and_then(|slot| *slot)
        .ok_or_else(|| invalid_input(format!("root node slot is not available: {root_node_id}")))?;

    Ok(ComputeSteps {
        waits,
        pops,
        body,
        typecast_inits,
        root_slot,
    })
}

fn scalar_binary_op(
    nodes: &[FusedEltwiseNode],
    op: FusedEltwiseOp,
    lhs: usize,
    rhs: usize,
) -> Option<(usize, u32, &'static str)> {
    let lhs_constant = constant_scalar_bits(nodes, lhs);
    let rhs_constant = constant_scalar_bits(nodes, rhs);
    match (op, lhs_constant, rhs_constant) {
        (FusedEltwiseOp::Add, None, Some(scalar)) => Some((lhs, scalar, "add_unary_tile")),
        (FusedEltwiseOp::Add, Some(scalar), None) => Some((rhs, scalar, "add_unary_tile")),
        (FusedEltwiseOp::Subtract, None, Some(scalar)) => Some((lhs, scalar, "sub_unary_tile")),
        (FusedEltwiseOp::Subtract, Some(scalar), None) => Some((rhs, scalar, "rsub_unary_tile")),
        (FusedEltwiseOp::Multiply, None, Some(scalar)) => Some((lhs, scalar, "mul_unary_tile")),
        (FusedEltwiseOp::Multiply, Some(scalar), None) => Some((rhs, scalar, "mul_unary_tile")),
        (FusedEltwiseOp::Divide, None, Some(scalar)) => Some((
            lhs,
            (1.0f32 / f32::from_bits(scalar)).to_bits(),
            "div_unary_tile",
        )),
        (FusedEltwiseOp::Max, None, Some(scalar)) => Some((lhs, scalar, "unary_max_tile")),
        (FusedEltwiseOp::Max, Some(scalar), None) => Some((rhs, scalar, "unary_max_tile")),
        _ => None,
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

fn is_supported_float(dtype: DType) -> bool {
    matches!(dtype, DType::Float16 | DType::Float16B | DType::Float32)
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
