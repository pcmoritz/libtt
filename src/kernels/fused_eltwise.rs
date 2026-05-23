use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer};
use crate::executable::{CompareDirection, FusedElementwiseKind};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, split_tile_range, Kernel, RuntimeArgsBuilder};
use std::fmt::Display;
use std::io;

const WRITER: &str = include_str!("../../kernels/tile_writer.cc");
const COMPUTE: &str = include_str!("../../kernels/fused_eltwise_compute.cc");
const MAX_FUSED_INPUTS: usize = 8;
const MAX_FUSED_NODES: usize = 16;

const HEADER_ADD_INT: &str = "compute_kernel_api/add_int_sfpu.h";
const HEADER_BINARY_MAX_MIN: &str = "compute_kernel_api/binary_max_min.h";
const HEADER_BINARY_SFPU: &str = "compute_kernel_api/eltwise_binary_sfpu.h";
const HEADER_BINOP_WITH_SCALAR: &str = "compute_kernel_api/eltwise_unary/binop_with_scalar.h";
const HEADER_COMP: &str = "compute_kernel_api/eltwise_unary/comp.h";
const HEADER_EXP: &str = "compute_kernel_api/eltwise_unary/exp.h";
const HEADER_MUL_INT: &str = "compute_kernel_api/mul_int_sfpu.h";
const HEADER_MUL_INT32: &str = "compute_kernel_api/mul_int32_sfpu.h";
const HEADER_NEGATIVE: &str = "compute_kernel_api/eltwise_unary/negative.h";
const HEADER_RDIV: &str = "compute_kernel_api/eltwise_unary/rdiv.h";
const HEADER_RPOW: &str = "compute_kernel_api/eltwise_unary/rpow.h";
const HEADER_RSQRT: &str = "compute_kernel_api/eltwise_unary/rsqrt.h";
const HEADER_SUB_INT: &str = "compute_kernel_api/sub_int_sfpu.h";
const HEADER_TRIGONOMETRY: &str = "compute_kernel_api/eltwise_unary/trigonometry.h";
const HEADER_TYPECAST: &str = "compute_kernel_api/eltwise_unary/typecast.h";

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

#[derive(Clone, Copy)]
struct UnaryCompute {
    header: &'static str,
    init: &'static str,
    tile: &'static str,
}

#[derive(Clone, Copy)]
struct BinaryCompute {
    header: &'static str,
    init: &'static str,
    tile: &'static str,
}

#[derive(Clone, Copy)]
enum ScalarOperand {
    Lhs,
    Rhs,
}

#[derive(Clone, Copy)]
struct ScalarCompute {
    operand: ScalarOperand,
    scalar: u32,
    header: Option<&'static str>,
    init: &'static str,
    tile: &'static str,
}

impl FusedEltwiseOp {
    fn arity(self) -> usize {
        match self {
            Self::Input | Self::Constant => 0,
            Self::Cosine
            | Self::Sine
            | Self::Negate
            | Self::Exponential
            | Self::Rsqrt
            | Self::Convert => 1,
            Self::Add
            | Self::Subtract
            | Self::Multiply
            | Self::Divide
            | Self::Power
            | Self::Max
            | Self::Compare(_) => 2,
        }
    }

    pub(crate) fn is_binary(self) -> bool {
        self.arity() == 2
    }

    pub(crate) fn is_compare(self) -> bool {
        matches!(self, Self::Compare(_))
    }

    pub(crate) fn binary_output_dtype(self, input_dtype: DType) -> io::Result<DType> {
        if !self.is_binary() {
            return Err(invalid_input(format!(
                "{self:?} is not a binary eltwise op"
            )));
        }
        self.validate_binary_dtype(input_dtype)?;
        Ok(if self.is_compare() {
            DType::UInt8
        } else {
            input_dtype
        })
    }

    fn validate_dtypes(
        self,
        node_index: usize,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> io::Result<()> {
        match self {
            Self::Input | Self::Constant => Ok(()),
            Self::Cosine | Self::Sine | Self::Negate | Self::Exponential | Self::Rsqrt => {
                let input_dtype = input_dtypes[0];
                validate_same_output_dtype(node_index, self, input_dtype, output_dtype)?;
                if !is_float_dtype(input_dtype) {
                    return Err(invalid_input(format!(
                        "node[{node_index}] {self:?} supports Float16, Float16B, and Float32 inputs, got {input_dtype:?}"
                    )));
                }
                Ok(())
            }
            Self::Convert => {
                let input_dtype = input_dtypes[0];
                if !is_convert_dtype(input_dtype) || !is_convert_dtype(output_dtype) {
                    return Err(invalid_input(format!(
                        "node[{node_index}] convert supports Float16B, Float32, Int32, UInt16, and UInt32, got {input_dtype:?} -> {output_dtype:?}"
                    )));
                }
                Ok(())
            }
            Self::Add
            | Self::Subtract
            | Self::Multiply
            | Self::Divide
            | Self::Power
            | Self::Max => {
                let input_dtype = self.validate_binary_input_dtypes(node_index, input_dtypes)?;
                validate_same_output_dtype(node_index, self, input_dtype, output_dtype)?;
                self.validate_binary_dtype(input_dtype)
                    .map_err(|err| invalid_input(format!("node[{node_index}] {err}")))
            }
            Self::Compare(_) => {
                let input_dtype = self.validate_binary_input_dtypes(node_index, input_dtypes)?;
                if output_dtype != DType::UInt8 {
                    return Err(invalid_input(format!(
                        "node[{node_index}] compare output dtype must be UInt8, got {output_dtype:?}"
                    )));
                }
                self.validate_binary_dtype(input_dtype)
                    .map_err(|err| invalid_input(format!("node[{node_index}] {err}")))
            }
        }
    }

    fn validate_binary_input_dtypes(
        self,
        node_index: usize,
        input_dtypes: &[DType],
    ) -> io::Result<DType> {
        let lhs = input_dtypes[0];
        let rhs = input_dtypes[1];
        if lhs != rhs {
            return Err(invalid_input(format!(
                "node[{node_index}] {self:?} input dtypes must match, got {lhs:?} and {rhs:?}"
            )));
        }
        Ok(lhs)
    }

    fn validate_binary_dtype(self, input_dtype: DType) -> io::Result<()> {
        let ok = match self {
            Self::Add | Self::Multiply => matches!(
                input_dtype,
                DType::Float16
                    | DType::Float16B
                    | DType::Float32
                    | DType::Int32
                    | DType::UInt16
                    | DType::UInt32
            ),
            Self::Subtract => matches!(
                input_dtype,
                DType::Float16 | DType::Float16B | DType::Float32 | DType::Int32
            ),
            Self::Divide | Self::Power | Self::Max => is_float_dtype(input_dtype),
            Self::Compare(_) => {
                matches!(input_dtype, DType::Float16B | DType::Float32 | DType::Int32)
            }
            _ => false,
        };
        if ok {
            Ok(())
        } else {
            Err(invalid_input(format!(
                "{self:?} does not support input dtype {input_dtype:?}"
            )))
        }
    }

    fn unary_compute(self) -> Option<UnaryCompute> {
        match self {
            Self::Negate => Some(UnaryCompute {
                header: HEADER_NEGATIVE,
                init: "negative_tile_init();",
                tile: "negative_tile(0);",
            }),
            Self::Cosine => Some(UnaryCompute {
                header: HEADER_TRIGONOMETRY,
                init: "cos_tile_init();",
                tile: "cos_tile(0);",
            }),
            Self::Sine => Some(UnaryCompute {
                header: HEADER_TRIGONOMETRY,
                init: "sin_tile_init();",
                tile: "sin_tile(0);",
            }),
            Self::Exponential => Some(UnaryCompute {
                header: HEADER_EXP,
                init: "exp_tile_init();",
                tile: "exp_tile(0);",
            }),
            Self::Rsqrt => Some(UnaryCompute {
                header: HEADER_RSQRT,
                init: "rsqrt_tile_init();",
                tile: "rsqrt_tile(0);",
            }),
            _ => None,
        }
    }

    fn data_format_binary_helper(self) -> Option<&'static str> {
        match self {
            Self::Add => Some("add_input"),
            Self::Subtract => Some("subtract_input"),
            Self::Multiply => Some("multiply_input"),
            _ => None,
        }
    }

    fn binary_compute(self) -> Option<BinaryCompute> {
        match self {
            Self::Divide => Some(BinaryCompute {
                header: HEADER_BINARY_SFPU,
                init: "div_binary_tile_init",
                tile: "div_binary_tile",
            }),
            Self::Power => Some(BinaryCompute {
                header: HEADER_BINARY_SFPU,
                init: "power_binary_tile_init",
                tile: "power_binary_tile",
            }),
            Self::Max => Some(BinaryCompute {
                header: HEADER_BINARY_MAX_MIN,
                init: "binary_max_tile_init",
                tile: "binary_max_tile",
            }),
            _ => None,
        }
    }

    fn scalar_compute(
        self,
        lhs_dtype: DType,
        rhs_dtype: DType,
        lhs_constant: Option<u32>,
        rhs_constant: Option<u32>,
    ) -> Option<ScalarCompute> {
        let scalar_op = |operand, dtype, scalar, float_tile, int32_tile| {
            let tile = match dtype {
                DType::Float16 | DType::Float16B | DType::Float32 => float_tile,
                DType::Int32 => int32_tile,
                _ => return None,
            };
            Some(ScalarCompute {
                operand,
                scalar,
                header: Some(HEADER_BINOP_WITH_SCALAR),
                init: "binop_with_scalar_tile_init",
                tile,
            })
        };
        let float_scalar_op = |operand, dtype, scalar, header, init, tile| {
            if !is_float_dtype(dtype) {
                return None;
            }
            Some(ScalarCompute {
                operand,
                scalar,
                header,
                init,
                tile,
            })
        };

        match (self, lhs_constant, rhs_constant) {
            (Self::Add, None, Some(scalar)) => scalar_op(
                ScalarOperand::Lhs,
                lhs_dtype,
                scalar,
                "add_unary_tile",
                "add_unary_tile_int32",
            ),
            (Self::Add, Some(scalar), None) => scalar_op(
                ScalarOperand::Rhs,
                rhs_dtype,
                scalar,
                "add_unary_tile",
                "add_unary_tile_int32",
            ),
            (Self::Subtract, None, Some(scalar)) => scalar_op(
                ScalarOperand::Lhs,
                lhs_dtype,
                scalar,
                "sub_unary_tile",
                "sub_unary_tile_int32",
            ),
            (Self::Subtract, Some(scalar), None) => float_scalar_op(
                ScalarOperand::Rhs,
                rhs_dtype,
                scalar,
                Some(HEADER_BINOP_WITH_SCALAR),
                "binop_with_scalar_tile_init",
                "rsub_unary_tile",
            ),
            (Self::Multiply, None, Some(scalar)) => float_scalar_op(
                ScalarOperand::Lhs,
                lhs_dtype,
                scalar,
                Some(HEADER_BINOP_WITH_SCALAR),
                "binop_with_scalar_tile_init",
                "mul_unary_tile",
            ),
            (Self::Multiply, Some(scalar), None) => float_scalar_op(
                ScalarOperand::Rhs,
                rhs_dtype,
                scalar,
                Some(HEADER_BINOP_WITH_SCALAR),
                "binop_with_scalar_tile_init",
                "mul_unary_tile",
            ),
            (Self::Divide, None, Some(scalar)) => float_scalar_op(
                ScalarOperand::Lhs,
                lhs_dtype,
                (1.0f32 / f32::from_bits(scalar)).to_bits(),
                Some(HEADER_BINOP_WITH_SCALAR),
                "binop_with_scalar_tile_init",
                "div_unary_tile",
            ),
            (Self::Divide, Some(scalar), None) => float_scalar_op(
                ScalarOperand::Rhs,
                rhs_dtype,
                scalar,
                Some(HEADER_RDIV),
                "rdiv_tile_init",
                "rdiv_tile",
            ),
            (Self::Power, None, Some(scalar)) => float_scalar_op(
                ScalarOperand::Lhs,
                lhs_dtype,
                scalar,
                None,
                "power_tile_init",
                "power_tile",
            ),
            (Self::Power, Some(scalar), None) => float_scalar_op(
                ScalarOperand::Rhs,
                rhs_dtype,
                scalar,
                Some(HEADER_RPOW),
                "rpow_tile_init",
                "rpow_tile",
            ),
            (Self::Max, None, Some(scalar)) => float_scalar_op(
                ScalarOperand::Lhs,
                lhs_dtype,
                scalar,
                None,
                "unary_max_tile_init",
                "unary_max_tile",
            ),
            (Self::Max, Some(scalar), None) => float_scalar_op(
                ScalarOperand::Rhs,
                rhs_dtype,
                scalar,
                None,
                "unary_max_tile_init",
                "unary_max_tile",
            ),
            _ => None,
        }
    }
}

impl From<FusedElementwiseKind> for FusedEltwiseOp {
    fn from(kind: FusedElementwiseKind) -> Self {
        match kind {
            FusedElementwiseKind::Input => Self::Input,
            FusedElementwiseKind::Constant => Self::Constant,
            FusedElementwiseKind::Add => Self::Add,
            FusedElementwiseKind::Subtract => Self::Subtract,
            FusedElementwiseKind::Multiply => Self::Multiply,
            FusedElementwiseKind::Divide => Self::Divide,
            FusedElementwiseKind::Max => Self::Max,
            FusedElementwiseKind::Negate => Self::Negate,
            FusedElementwiseKind::Exponential => Self::Exponential,
            FusedElementwiseKind::Rsqrt => Self::Rsqrt,
        }
    }
}

impl CompareDirection {
    fn reversed(self) -> Self {
        match self {
            Self::Eq => Self::Eq,
            Self::Ne => Self::Ne,
            Self::Ge => Self::Le,
            Self::Gt => Self::Lt,
            Self::Le => Self::Ge,
            Self::Lt => Self::Gt,
        }
    }

    fn variant(self) -> &'static str {
        match self {
            Self::Eq => "Eq",
            Self::Ne => "Ne",
            Self::Ge => "Ge",
            Self::Gt => "Gt",
            Self::Le => "Le",
            Self::Lt => "Lt",
        }
    }

    fn unary_init(self) -> &'static str {
        match self {
            Self::Eq => "unary_eq_tile_init",
            Self::Ne => "unary_ne_tile_init",
            Self::Ge => "unary_ge_tile_init",
            Self::Gt => "unary_gt_tile_init",
            Self::Le => "unary_le_tile_init",
            Self::Lt => "unary_lt_tile_init",
        }
    }

    fn unary_tile(self, int32_input: bool) -> &'static str {
        match (self, int32_input) {
            (Self::Eq, false) => "unary_eq_tile",
            (Self::Ne, false) => "unary_ne_tile",
            (Self::Ge, false) => "unary_ge_tile",
            (Self::Gt, false) => "unary_gt_tile",
            (Self::Le, false) => "unary_le_tile",
            (Self::Lt, false) => "unary_lt_tile",
            (Self::Eq, true) => "unary_eq_tile_int32",
            (Self::Ne, true) => "unary_ne_tile_int32",
            (Self::Ge, true) => "unary_ge_tile_int32",
            (Self::Gt, true) => "unary_gt_tile_int32",
            (Self::Le, true) => "unary_le_tile_int32",
            (Self::Lt, true) => "unary_lt_tile_int32",
        }
    }
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

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct FusedEltwiseProgramKey {
    cores: Vec<CoreCoord>,
    tile_count: u32,
    output_dtype: DType,
    nodes: Vec<FusedEltwiseNode>,
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
    external_inputs: &[&DramBuffer],
    nodes: &[FusedEltwiseNode],
    shape: &[usize],
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    let input_reads = validate_and_collect_inputs(external_inputs, nodes, shape)?;

    let output_tiles = tiled_shape_tile_count(shape)?;
    let tile_count = u32_arg(output_tiles, "tile count")?;
    let cores = select_worker_cores(device.cores_ref(), output_tiles)?;
    let output_dtype = nodes[nodes.len() - 1].dtype;
    let output_shape = tiled_allocation_shape(shape)?;
    let output = device.alloc(output_tiles, output_dtype, &output_shape, name)?;

    let mut input_addrs = Vec::with_capacity(input_reads.len());
    for (index, &input) in input_reads.iter().enumerate() {
        input_addrs.push(u32_arg(input.addr, &format!("input[{index}] address"))?);
    }

    let kernel = FusedEltwiseKernel {
        input_addrs,
        output_addr: u32_arg(output.addr, "output address")?,
        key: FusedEltwiseProgramKey {
            cores,
            tile_count,
            output_dtype,
            nodes: nodes.to_vec(),
        },
    };
    kernel.run(device)?;
    Ok(output)
}

fn validate_and_collect_inputs<'a>(
    external_inputs: &[&'a DramBuffer],
    nodes: &[FusedEltwiseNode],
    shape: &[usize],
) -> io::Result<Vec<&'a DramBuffer>> {
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
    let root = nodes
        .last()
        .expect("nodes was already checked as non-empty");
    if matches!(root.op, FusedEltwiseOp::Input | FusedEltwiseOp::Constant) {
        return Err(invalid_input(format!(
            "fused eltwise root node must be the final operation, got {:?}",
            root.op
        )));
    }

    let expected_tiles = tiled_shape_tile_count(shape)?;
    let expected_shape = tiled_allocation_shape(shape)?;
    let mut input_reads = Vec::new();
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
                let buffer = external_inputs[input_index];
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
                input_reads.push(buffer);
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
            _ => validate_node_inputs(index, node, node.op.arity())?,
        }
        for &input_node in &node.input_nodes {
            if usize::try_from(input_node).map_or(true, |input| input >= index) {
                return Err(invalid_input(format!(
                    "node[{index}] references non-prior input node {input_node}"
                )));
            }
        }
        let input_dtypes = node
            .input_nodes
            .iter()
            .map(|&input_node| nodes[input_node as usize].dtype)
            .collect::<Vec<_>>();
        node.op.validate_dtypes(index, &input_dtypes, node.dtype)?;
    }
    if input_reads.is_empty() || input_reads.len() > MAX_FUSED_INPUTS {
        return Err(invalid_input(format!(
            "fused eltwise requires 1..={MAX_FUSED_INPUTS} leaf inputs, got {}",
            input_reads.len()
        )));
    }
    Ok(input_reads)
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
    let input_nodes = fused_input_nodes(&key.nodes);
    let input_count = input_nodes.len();
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

    let (_, intermediate_cbs) = cb_plan(&key.nodes)?;
    let mut cbs = Vec::with_capacity(input_count + intermediate_cbs.len() + 1);
    for (index, node) in input_nodes.iter().enumerate() {
        cbs.push(CBConfig::new(index, node.dtype));
    }
    for (cb, dtype) in intermediate_cbs {
        cbs.push(CBConfig::new(cb as usize, dtype));
    }
    cbs.push(CBConfig::new(16, key.output_dtype));

    let dst_accum_mode = key
        .nodes
        .iter()
        .map(|node| &node.dtype)
        .chain(std::iter::once(&key.output_dtype))
        .any(|dtype| matches!(dtype, DType::Float32 | DType::Int32 | DType::UInt32));

    Ok(Program {
        reader_kernel: reader_source(&input_nodes),
        compute_kernel: compute_source(&key)?,
        writer_kernel: WRITER.to_owned(),
        compile: CompileConfig {
            cbs,
            dst_accum_mode,
            ..CompileConfig::default()
        },
        name: format!("fused_eltwise_{}_{}", input_count, key.nodes.len()),
        ..Program::new(runtime_args)
    })
}

fn fused_input_nodes(nodes: &[FusedEltwiseNode]) -> Vec<&FusedEltwiseNode> {
    nodes
        .iter()
        .filter(|node| node.op == FusedEltwiseOp::Input)
        .collect()
}

fn reader_source(input_nodes: &[&FusedEltwiseNode]) -> String {
    let input_count = input_nodes.len();
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
        let tile_id = if input_nodes[index].single_tile_broadcast {
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
        if input_nodes[index].single_tile_broadcast {
            let mode = match input_nodes[index].dtype {
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
    let steps = compute_steps(&key.nodes)?;
    Ok(COMPUTE
        .replace("FUSED_HEADERS", &steps.features.headers_source())
        .replace("FUSED_HELPERS", &steps.features.helpers_source())
        .replace("FUSED_TYPECAST_INITS", &steps.typecast_inits)
        .replace("FUSED_STEPS", &steps.body))
}

#[derive(Default)]
struct ComputeSourceFeatures {
    headers: Vec<&'static str>,
    add_input_helper: bool,
    subtract_input_helper: bool,
    multiply_input_helper: bool,
    compare_helpers: bool,
}

impl ComputeSourceFeatures {
    fn add_header(&mut self, header: &'static str) {
        if !self.headers.contains(&header) {
            self.headers.push(header);
        }
    }

    fn add_unary(&mut self, unary: UnaryCompute) {
        self.add_header(unary.header);
    }

    fn add_binary(&mut self, binary: BinaryCompute) {
        self.add_header(binary.header);
    }

    fn add_scalar(&mut self, scalar: ScalarCompute) {
        if let Some(header) = scalar.header {
            self.add_header(header);
        }
    }

    fn add_typecast(&mut self) {
        self.add_header(HEADER_TYPECAST);
    }

    fn add_unary_compare(&mut self) {
        self.add_header(HEADER_COMP);
    }

    fn add_compare_helpers(&mut self) {
        self.compare_helpers = true;
        self.add_header(HEADER_BINARY_SFPU);
        self.add_header(HEADER_SUB_INT);
        self.add_header(HEADER_COMP);
    }

    fn add_data_format_binary_helper(&mut self, op: FusedEltwiseOp) {
        self.add_header(HEADER_BINARY_SFPU);
        match op {
            FusedEltwiseOp::Add => {
                self.add_input_helper = true;
                self.add_header(HEADER_ADD_INT);
            }
            FusedEltwiseOp::Subtract => {
                self.subtract_input_helper = true;
                self.add_header(HEADER_SUB_INT);
            }
            FusedEltwiseOp::Multiply => {
                self.multiply_input_helper = true;
                self.add_header(HEADER_MUL_INT);
                self.add_header(HEADER_MUL_INT32);
            }
            _ => {}
        }
    }

    fn headers_source(&self) -> String {
        self.headers
            .iter()
            .map(|header| format!("#include \"{header}\"\n"))
            .collect()
    }

    fn helpers_source(&self) -> String {
        let mut helpers = String::new();
        if self.add_input_helper || self.subtract_input_helper || self.multiply_input_helper {
            helpers.push_str(binary_input_data_format_helper());
        }
        if self.add_input_helper {
            helpers.push_str(add_input_helper());
        }
        if self.subtract_input_helper {
            helpers.push_str(subtract_input_helper());
        }
        if self.multiply_input_helper {
            helpers.push_str(multiply_input_helper());
        }
        if self.compare_helpers {
            helpers.push_str(compare_helpers());
        }
        helpers
    }
}

fn binary_input_data_format_helper() -> &'static str {
    r#"
constexpr DataFormat binary_input_data_format(uint32_t cb_lhs, uint32_t cb_out) {
#ifdef UCK_CHLKC_PACK
  return static_cast<DataFormat>((uint)pack_src_format[cb_out]);
#else
  return static_cast<DataFormat>((uint)unpack_src_format[cb_lhs]);
#endif
}

"#
}

fn add_input_helper() -> &'static str {
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

"#
}

fn subtract_input_helper() -> &'static str {
    r#"
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

"#
}

fn multiply_input_helper() -> &'static str {
    r#"
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

"#
}

fn compare_helpers() -> &'static str {
    r#"
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
    features: ComputeSourceFeatures,
}

fn compute_steps(nodes: &[FusedEltwiseNode]) -> io::Result<ComputeSteps> {
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

    let (node_cbs, _) = cb_plan(nodes)?;
    let mut body = String::new();
    let mut typecast_inits = Vec::<String>::new();
    let mut features = ComputeSourceFeatures::default();

    for (index, node) in nodes.iter().enumerate() {
        match node.op.arity() {
            0 => {}
            1 => {
                let input = node.input_nodes[0] as usize;
                let input_cb = cb_for_node(&node_cbs, input)?;
                let output_cb = cb_for_node(&node_cbs, index)?;
                append_waits(&mut body, &[input_cb]);
                let unary = node.op.unary_compute();
                let init = unary.map_or("", |op| op.init);
                body.push_str(&format!(
                    "    {init}\n    cb_reserve_back(tt::CBIndex::c_{output_cb}, 1);\n    tile_regs_acquire();\n    copy_tile_to_dst_init_short(tt::CBIndex::c_{input_cb});\n    copy_tile(tt::CBIndex::c_{input_cb}, 0, 0);\n"
                ));
                if let Some(unary) = unary {
                    features.add_unary(unary);
                    body.push_str(&format!("    {}\n", unary.tile));
                } else {
                    debug_assert_eq!(node.op, FusedEltwiseOp::Convert);
                    features.add_typecast();
                    let from = nodes[input].dtype as u32;
                    let to = node.dtype as u32;
                    let init = format!("  typecast_tile_init<{from}, {to}>();\n");
                    if !typecast_inits.contains(&init) {
                        typecast_inits.push(init);
                    }
                    body.push_str(&format!("    typecast_tile<{from}, {to}>(0);\n"));
                }
                append_pack_and_pop(
                    &mut body,
                    output_cb,
                    &[input],
                    &node_cbs,
                    &mut remaining_uses,
                )?;
            }
            2 => {
                let lhs = node.input_nodes[0] as usize;
                let rhs = node.input_nodes[1] as usize;
                if let FusedEltwiseOp::Compare(direction) = node.op {
                    if let Some((value_node, scalar, scalar_direction)) =
                        scalar_compare_op(nodes, lhs, rhs, direction)
                    {
                        let value_cb = cb_for_node(&node_cbs, value_node)?;
                        let output_cb = cb_for_node(&node_cbs, index)?;
                        let int32_input = nodes[value_node].dtype == DType::Int32;
                        let init = scalar_direction.unary_init();
                        let call = scalar_direction.unary_tile(int32_input);
                        append_waits(&mut body, &[value_cb]);
                        features.add_unary_compare();
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
                if let Some(scalar_op) = node.op.scalar_compute(
                    nodes[lhs].dtype,
                    nodes[rhs].dtype,
                    constant_scalar_bits(nodes, lhs),
                    constant_scalar_bits(nodes, rhs),
                ) {
                    let value_node = match scalar_op.operand {
                        ScalarOperand::Lhs => lhs,
                        ScalarOperand::Rhs => rhs,
                    };
                    let value_cb = cb_for_node(&node_cbs, value_node)?;
                    let output_cb = cb_for_node(&node_cbs, index)?;
                    append_waits(&mut body, &[value_cb]);
                    features.add_scalar(scalar_op);
                    body.push_str(&format!(
                        "    {}();\n    cb_reserve_back(tt::CBIndex::c_{output_cb}, 1);\n    tile_regs_acquire();\n    copy_tile_to_dst_init_short(tt::CBIndex::c_{value_cb});\n    copy_tile(tt::CBIndex::c_{value_cb}, 0, 0);\n    {}(0, {});\n",
                        scalar_op.init, scalar_op.tile, scalar_op.scalar
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
                    let direction = direction.variant();
                    append_waits(&mut body, &[lhs_cb, rhs_cb]);
                    features.add_compare_helpers();
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
                if let Some(helper) = node.op.data_format_binary_helper() {
                    append_waits(&mut body, &[lhs_cb, rhs_cb]);
                    features.add_data_format_binary_helper(node.op);
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
                let binary = node.op.binary_compute().ok_or_else(|| {
                    invalid_input(format!("missing binary lowering for {:?}", node.op))
                })?;
                append_waits(&mut body, &[lhs_cb, rhs_cb]);
                features.add_binary(binary);
                body.push_str(&format!(
                    "    {}();\n    cb_reserve_back(tt::CBIndex::c_{output_cb}, 1);\n    tile_regs_acquire();\n    copy_tile_to_dst_init_short_with_dt(tt::CBIndex::c_{rhs_cb}, tt::CBIndex::c_{lhs_cb});\n    copy_tile(tt::CBIndex::c_{lhs_cb}, 0, 0);\n    copy_tile_to_dst_init_short_with_dt(tt::CBIndex::c_{lhs_cb}, tt::CBIndex::c_{rhs_cb});\n    copy_tile(tt::CBIndex::c_{rhs_cb}, 0, 1);\n    {}(0, 1, 0);\n",
                    binary.init, binary.tile
                ));
                append_pack_and_pop(
                    &mut body,
                    output_cb,
                    &[lhs, rhs],
                    &node_cbs,
                    &mut remaining_uses,
                )?;
            }
            _ => unreachable!("fused eltwise op arity is limited to 0, 1, or 2"),
        }
    }

    Ok(ComputeSteps {
        body,
        typecast_inits: typecast_inits.concat(),
        features,
    })
}

fn cb_plan(nodes: &[FusedEltwiseNode]) -> io::Result<(Vec<Option<u32>>, Vec<(u32, DType)>)> {
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

    let root_index = nodes
        .len()
        .checked_sub(1)
        .ok_or_else(|| invalid_input("fused eltwise requires at least one node"))?;
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
        (Some(scalar), None) => Some((rhs, scalar, direction.reversed())),
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

fn bool_literal(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
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

fn validate_same_output_dtype(
    node_index: usize,
    op: FusedEltwiseOp,
    input_dtype: DType,
    output_dtype: DType,
) -> io::Result<()> {
    if output_dtype == input_dtype {
        return Ok(());
    }
    Err(invalid_input(format!(
        "node[{node_index}] {op:?} output dtype must match input dtype, got {input_dtype:?} -> {output_dtype:?}"
    )))
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

    fn program_key(nodes: Vec<FusedEltwiseNode>) -> FusedEltwiseProgramKey {
        FusedEltwiseProgramKey {
            cores: Vec::new(),
            tile_count: 1,
            output_dtype: nodes.last().expect("test nodes must not be empty").dtype,
            nodes,
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
        let steps = compute_steps(&nodes).expect("constant / value should lower");

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
        let steps = compute_steps(&nodes).expect("value ** constant should lower");

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
        let steps = compute_steps(&nodes).expect("value > constant should lower");

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
        let steps = compute_steps(&nodes).expect("constant > value should lower");

        assert!(steps.body.contains("unary_lt_tile_init();"));
        assert!(steps.body.contains("unary_lt_tile(0, 0);"));
    }

    #[test]
    fn compute_steps_handles_constant_right_max_as_unary_max() {
        let mut constant = node(FusedEltwiseOp::Constant, Vec::new());
        constant.packed_value = 0;

        let nodes = vec![
            node(FusedEltwiseOp::Input, Vec::new()),
            constant,
            node(FusedEltwiseOp::Max, vec![0, 1]),
        ];
        let steps = compute_steps(&nodes).expect("max(value, constant) should lower");

        assert!(steps.body.contains("unary_max_tile_init();"));
        assert!(steps.body.contains("unary_max_tile(0, 0);"));
    }

    #[test]
    fn compute_source_only_emits_used_add_helpers() {
        let nodes = vec![
            node(FusedEltwiseOp::Input, Vec::new()),
            node(FusedEltwiseOp::Input, Vec::new()),
            node(FusedEltwiseOp::Add, vec![0, 1]),
        ];
        let source = compute_source(&program_key(nodes)).expect("add source should generate");

        assert!(source.contains(HEADER_BINARY_SFPU));
        assert!(source.contains(HEADER_ADD_INT));
        assert!(source.contains("ALWI void add_input_tile"));
        assert!(!source.contains(HEADER_EXP));
        assert!(!source.contains(HEADER_COMP));
        assert!(!source.contains("ALWI void compare_zero_tile"));
        assert!(!source.contains("FUSED_"));
    }

    #[test]
    fn compute_source_only_emits_compare_helpers_for_compare() {
        let nodes = vec![
            node(FusedEltwiseOp::Input, Vec::new()),
            node(FusedEltwiseOp::Input, Vec::new()),
            node(FusedEltwiseOp::Compare(CompareDirection::Gt), vec![0, 1]),
        ];
        let source = compute_source(&program_key(nodes)).expect("compare source should generate");

        assert!(source.contains(HEADER_COMP));
        assert!(source.contains(HEADER_SUB_INT));
        assert!(source.contains("ALWI void compare_zero_tile"));
        assert!(!source.contains(HEADER_ADD_INT));
        assert!(!source.contains("ALWI void add_input_tile"));
        assert!(!source.contains("FUSED_"));
    }
}
