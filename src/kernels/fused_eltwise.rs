use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{
    tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer, TILE_C, TILE_R,
};
use crate::executable::{CompareDirection, FusedElementwiseKind, FusedElementwiseNode};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, split_tile_range, Kernel, RuntimeArgsBuilder};
use crate::utils::pjrt_buffer_type_to_dtype;
use std::fmt::{Display, Write};
use std::io;

const WRITER: &str = include_str!("../../kernels/tile_writer.cc");
const COMPUTE: &str = include_str!("../../kernels/fused_eltwise_compute.cc");
const HELPER_BINARY_INPUT_DATA_FORMAT: &str =
    include_str!("../../kernels/fused_eltwise_helpers/binary_input_data_format.cc.inc");
const HELPER_ADD_INPUT: &str = include_str!("../../kernels/fused_eltwise_helpers/add_input.cc.inc");
const HELPER_SUBTRACT_INPUT: &str =
    include_str!("../../kernels/fused_eltwise_helpers/subtract_input.cc.inc");
const HELPER_MULTIPLY_INPUT: &str =
    include_str!("../../kernels/fused_eltwise_helpers/multiply_input.cc.inc");
const HELPER_COMPARE: &str = include_str!("../../kernels/fused_eltwise_helpers/compare.cc.inc");
pub(crate) const MAX_FUSED_INPUTS: usize = 8;
pub(crate) const MAX_FUSED_NODES: usize = 16;

const HEADER_ADD_INT: &str = "compute_kernel_api/add_int_sfpu.h";
const HEADER_BINARY_MAX_MIN: &str = "compute_kernel_api/binary_max_min.h";
const HEADER_BINARY_SFPU: &str = "compute_kernel_api/eltwise_binary_sfpu.h";
const HEADER_BINOP_WITH_SCALAR: &str = "compute_kernel_api/eltwise_unary/binop_with_scalar.h";
const HEADER_COMP: &str = "compute_kernel_api/eltwise_unary/comp.h";
const HEADER_EXP: &str = "compute_kernel_api/eltwise_unary/exp.h";
const HEADER_LOG: &str = "compute_kernel_api.h";
const HEADER_MUL_INT: &str = "compute_kernel_api/mul_int_sfpu.h";
const HEADER_MUL_INT32: &str = "compute_kernel_api/mul_int32_sfpu.h";
const HEADER_NEGATIVE: &str = "compute_kernel_api/eltwise_unary/negative.h";
const HEADER_RDIV: &str = "compute_kernel_api/eltwise_unary/rdiv.h";
const HEADER_RPOW: &str = "compute_kernel_api/eltwise_unary/rpow.h";
const HEADER_RSQRT: &str = "compute_kernel_api/eltwise_unary/rsqrt.h";
const HEADER_SFPU_SPLIT: &str = "compute_kernel_api/eltwise_unary/sfpu_split_includes.h";
const HEADER_SUB_INT: &str = "compute_kernel_api/sub_int_sfpu.h";
const HEADER_TRIGONOMETRY: &str = "compute_kernel_api/eltwise_unary/trigonometry.h";
const HEADER_TYPECAST: &str = "compute_kernel_api/eltwise_unary/typecast.h";

const HELPER_SELECT: &str = r#"
#ifdef TRISC_MATH
#define FUSED_SELECT_ITERATIONS (8)

template <bool KeepWhenPred, bool Int32Value>
inline void fused_select_gate_value(const uint dst_index_pred, const uint dst_index_value, const uint dst_index_out) {
  constexpr uint dst_tile_size_sfpi = 32;
  for (int i = 0; i < FUSED_SELECT_ITERATIONS; ++i) {
    vInt pred = dst_reg[dst_index_pred * dst_tile_size_sfpi];
    if constexpr (Int32Value) {
      vInt values = dst_reg[dst_index_value * dst_tile_size_sfpi];
      if constexpr (KeepWhenPred) {
        v_if (pred != 0) {
          dst_reg[dst_index_out * dst_tile_size_sfpi] = values;
        } v_else {
          dst_reg[dst_index_out * dst_tile_size_sfpi] = 0;
        } v_endif;
      } else {
        v_if (pred != 0) {
          dst_reg[dst_index_out * dst_tile_size_sfpi] = 0;
        } v_else {
          dst_reg[dst_index_out * dst_tile_size_sfpi] = values;
        } v_endif;
      }
    } else {
      vFloat values = dst_reg[dst_index_value * dst_tile_size_sfpi];
      if constexpr (KeepWhenPred) {
        v_if (pred != 0) {
          dst_reg[dst_index_out * dst_tile_size_sfpi] = values;
        } v_else {
          dst_reg[dst_index_out * dst_tile_size_sfpi] = 0.0f;
        } v_endif;
      } else {
        v_if (pred != 0) {
          dst_reg[dst_index_out * dst_tile_size_sfpi] = 0.0f;
        } v_else {
          dst_reg[dst_index_out * dst_tile_size_sfpi] = values;
        } v_endif;
      }
    }
    dst_reg++;
  }
}
#endif

constexpr DataFormat fused_select_value_format(uint32_t cb_value, uint32_t cb_out) {
#ifdef UCK_CHLKC_PACK
  return static_cast<DataFormat>((uint)pack_src_format[cb_out]);
#else
  return static_cast<DataFormat>((uint)unpack_src_format[cb_value]);
#endif
}

template <DataFormat Format>
ALWI void fused_select_add_init() {
  if constexpr (Format == DataFormat::Int32) {
    add_int_tile_init();
  } else {
    add_binary_tile_init();
  }
}

template <DataFormat Format>
ALWI void fused_select_add_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
  if constexpr (Format == DataFormat::Int32) {
    add_int32_tile(idst0, idst1, odst);
  } else {
    add_binary_tile(idst0, idst1, odst);
  }
}
"#;

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

impl FusedElementwiseKind {
    fn arity(self) -> usize {
        match self {
            Self::Input | Self::Constant => 0,
            Self::Cosine
            | Self::Sine
            | Self::Negate
            | Self::Exponential
            | Self::Log
            | Self::Rsqrt
            | Self::Convert => 1,
            Self::Add
            | Self::Subtract
            | Self::Multiply
            | Self::Divide
            | Self::Power
            | Self::Max
            | Self::Compare(_) => 2,
            Self::Select => 3,
        }
    }

    fn validate_dtypes(
        self,
        node_index: usize,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> io::Result<()> {
        match self {
            Self::Input | Self::Constant => Ok(()),
            Self::Cosine
            | Self::Sine
            | Self::Negate
            | Self::Exponential
            | Self::Log
            | Self::Rsqrt => {
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
                if !is_supported_convert_dtype(input_dtype)
                    || !is_supported_convert_dtype(output_dtype)
                {
                    return Err(invalid_input(format!(
                        "node[{node_index}] convert supports Float16, Float16B, Float32, Int32, UInt8, UInt16, and UInt32, got {input_dtype:?} -> {output_dtype:?}"
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
            Self::Select => {
                let pred_dtype = input_dtypes[0];
                let true_dtype = input_dtypes[1];
                let false_dtype = input_dtypes[2];
                if pred_dtype != DType::UInt8 {
                    return Err(invalid_input(format!(
                        "node[{node_index}] select predicate dtype must be UInt8, got {pred_dtype:?}"
                    )));
                }
                if true_dtype != false_dtype {
                    return Err(invalid_input(format!(
                        "node[{node_index}] select value dtypes must match, got {true_dtype:?} and {false_dtype:?}"
                    )));
                }
                validate_same_output_dtype(node_index, self, true_dtype, output_dtype)?;
                if !is_supported_select_value_dtype(true_dtype) {
                    return Err(invalid_input(format!(
                        "node[{node_index}] select supports Float16, Float16B, Float32, and Int32 values, got {true_dtype:?}"
                    )));
                }
                Ok(())
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
        match (self, input_dtype) {
            (
                Self::Add | Self::Multiply,
                DType::Float16
                | DType::Float16B
                | DType::Float32
                | DType::Int32
                | DType::UInt16
                | DType::UInt32,
            )
            | (
                Self::Subtract | Self::Compare(_),
                DType::Float16 | DType::Float16B | DType::Float32 | DType::Int32,
            )
            | (
                Self::Divide | Self::Power | Self::Max,
                DType::Float16 | DType::Float16B | DType::Float32,
            ) => Ok(()),
            _ => Err(invalid_input(format!(
                "{self:?} does not support input dtype {input_dtype:?}"
            ))),
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
            Self::Log => Some(UnaryCompute {
                header: HEADER_LOG,
                init: "log_tile_init();",
                tile: "log_tile(0);",
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
struct FusedEltwiseProgramKey {
    cores: Vec<CoreCoord>,
    tile_offset: u32,
    tile_count: u32,
    output_shape: Vec<u32>,
    output_dtype: DType,
    input_specs: Vec<FusedInputSpec>,
    nodes: Vec<FusedElementwiseNode>,
}

struct FusedEltwiseKernel {
    input_addrs: Vec<u32>,
    output_addr: u32,
    key: FusedEltwiseProgramKey,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct FusedInputSpec {
    shape: Vec<u32>,
    broadcast_dimensions: Vec<u32>,
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

fn node_dtype(node: &FusedElementwiseNode) -> io::Result<DType> {
    pjrt_buffer_type_to_dtype(node.element_type)
}

fn fused_input_specs(
    external_input_shapes: &[Vec<usize>],
    nodes: &[FusedElementwiseNode],
    output_shape: &[usize],
) -> io::Result<Vec<FusedInputSpec>> {
    let mut broadcast_dimensions = vec![None::<Vec<i64>>; external_input_shapes.len()];
    for node in nodes {
        if node.kind != FusedElementwiseKind::Input || node.broadcast_dimensions.is_empty() {
            continue;
        }
        let input_index = node.input_index as usize;
        let Some(slot) = broadcast_dimensions.get_mut(input_index) else {
            return Err(invalid_input(format!(
                "input index {} is out of bounds for {} inputs",
                node.input_index,
                external_input_shapes.len()
            )));
        };
        if let Some(existing) = slot {
            if existing != &node.broadcast_dimensions {
                return Err(invalid_input(format!(
                    "input[{input_index}] has conflicting fused broadcast dimensions"
                )));
            }
        } else {
            *slot = Some(node.broadcast_dimensions.clone());
        }
    }

    external_input_shapes
        .iter()
        .enumerate()
        .map(|(index, shape)| {
            let dims = broadcast_dimensions[index].clone().unwrap_or_default();
            if !dims.is_empty() && !supports_input_broadcast(shape, output_shape, &dims) {
                return Err(invalid_input(format!(
                    "input[{index}] broadcast {:?} -> {:?} dims {:?} is not supported by fused eltwise",
                    shape, output_shape, dims
                )));
            }
            Ok(FusedInputSpec {
                shape: u32_shape(shape, "fused eltwise input shape")?,
                broadcast_dimensions: u32_dims(&dims, "fused eltwise broadcast dimensions")?,
            })
        })
        .collect()
}

pub(crate) fn eltwise(
    device: &mut Device,
    external_inputs: &[&DramBuffer],
    external_input_shapes: &[Vec<usize>],
    nodes: &[FusedElementwiseNode],
    shape: &[usize],
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    let input_specs = fused_input_specs(external_input_shapes, nodes, shape)?;
    let input_reads = validate_and_collect_inputs(external_inputs, &input_specs, nodes, shape)?;

    let output_tiles = tiled_shape_tile_count(shape)?;
    let output_dtype = node_dtype(&nodes[nodes.len() - 1])?;
    let output_shape = tiled_allocation_shape(shape)?;
    let output = device.alloc(output_tiles, output_dtype, &output_shape, name)?;

    let mut input_addrs = Vec::with_capacity(input_reads.len());
    for (index, &input) in input_reads.iter().enumerate() {
        input_addrs.push(u32_arg(input.addr, &format!("input[{index}] address"))?);
    }

    let output_addr = u32_arg(output.addr, "output address")?;
    let output_shape = u32_shape(shape, "fused eltwise output shape")?;
    let max_tiles_per_launch = device.cores_ref().len().max(1);
    let mut tile_offset = 0usize;
    while tile_offset < output_tiles {
        let chunk_tiles = (output_tiles - tile_offset).min(max_tiles_per_launch);
        let cores = select_worker_cores(device.cores_ref(), chunk_tiles)?;
        let kernel = FusedEltwiseKernel {
            input_addrs: input_addrs.clone(),
            output_addr,
            key: FusedEltwiseProgramKey {
                cores,
                tile_offset: u32_arg(tile_offset, "tile offset")?,
                tile_count: u32_arg(chunk_tiles, "tile count")?,
                output_shape: output_shape.clone(),
                output_dtype,
                input_specs: input_specs.clone(),
                nodes: nodes.to_vec(),
            },
        };
        kernel.run(device)?;
        tile_offset += chunk_tiles;
    }
    Ok(output)
}

fn validate_and_collect_inputs<'a>(
    external_inputs: &[&'a DramBuffer],
    input_specs: &[FusedInputSpec],
    nodes: &[FusedElementwiseNode],
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
    if matches!(
        root.kind,
        FusedElementwiseKind::Input | FusedElementwiseKind::Constant
    ) {
        return Err(invalid_input(format!(
            "fused eltwise root node must be the final operation, got {:?}",
            root.kind
        )));
    }

    let expected_tiles = tiled_shape_tile_count(shape)?;
    let expected_shape = tiled_allocation_shape(shape)?;
    let mut input_reads = Vec::new();
    for (index, node) in nodes.iter().enumerate() {
        let dtype = node_dtype(node)?;
        match node.kind {
            FusedElementwiseKind::Input => {
                if !is_supported_convert_dtype(dtype) {
                    return Err(invalid_input(format!(
                        "node[{index}] input dtype {:?} is not supported by fused eltwise",
                        dtype
                    )));
                }
                let input_index = node.input_index as usize;
                if input_index >= external_inputs.len() {
                    return Err(invalid_input(format!(
                        "node[{index}] input index {} is out of bounds for {} inputs",
                        node.input_index,
                        external_inputs.len()
                    )));
                }
                let input_spec = input_specs.get(input_index).ok_or_else(|| {
                    invalid_input(format!(
                        "input[{input_index}] is missing a fused input spec"
                    ))
                })?;
                let buffer = external_inputs[input_index];
                let input_dtype = buffer.dtype;
                if input_dtype != dtype {
                    return Err(invalid_input(format!(
                        "node[{index}] input dtype mismatch: node {:?}, input {:?}",
                        dtype, input_dtype
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
                } else if !node.broadcast_dimensions.is_empty() {
                    let input_shape = usize_shape(&input_spec.shape)?;
                    let expected_input_shape = tiled_allocation_shape(&input_shape)?;
                    if buffer.shape != expected_input_shape {
                        return Err(invalid_input(format!(
                            "node[{index}] broadcast input allocation shape mismatch: got {:?}, expected {:?} for logical shape {:?}",
                            buffer.shape, expected_input_shape, input_shape
                        )));
                    }
                    let expected_input_tiles = tiled_shape_tile_count(&input_shape)?;
                    if buffer.num_tiles != expected_input_tiles {
                        return Err(invalid_input(format!(
                            "node[{index}] broadcast input tile count mismatch: got {}, expected {expected_input_tiles}",
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
            FusedElementwiseKind::Constant => {
                if !is_supported_convert_dtype(dtype) {
                    return Err(invalid_input(format!(
                        "node[{index}] constant dtype {:?} is not supported by fused eltwise",
                        dtype
                    )));
                }
                if !node.input_nodes.is_empty() {
                    return Err(invalid_input(format!(
                        "node[{index}] constant node must not have operands"
                    )));
                }
            }
            _ => {
                let expected = node.kind.arity();
                if node.input_nodes.len() != expected {
                    return Err(invalid_input(format!(
                        "node[{index}] {:?} expected {expected} operands, got {}",
                        node.kind,
                        node.input_nodes.len()
                    )));
                }
            }
        }
        for &input_node in &node.input_nodes {
            if input_node as usize >= index {
                return Err(invalid_input(format!(
                    "node[{index}] references non-prior input node {input_node}"
                )));
            }
        }
        let input_dtypes = node
            .input_nodes
            .iter()
            .map(|&input_node| node_dtype(&nodes[input_node as usize]))
            .collect::<io::Result<Vec<_>>>()?;
        node.kind.validate_dtypes(index, &input_dtypes, dtype)?;
    }
    if input_reads.is_empty() || input_reads.len() > MAX_FUSED_INPUTS {
        return Err(invalid_input(format!(
            "fused eltwise requires 1..={MAX_FUSED_INPUTS} leaf inputs, got {}",
            input_reads.len()
        )));
    }
    Ok(input_reads)
}

fn fused_eltwise_program(key: FusedEltwiseProgramKey) -> io::Result<Program> {
    let (node_cbs, intermediate_cbs) = cb_plan(&key.nodes)?;
    let leaf_nodes = fused_leaf_nodes(&key.nodes, &node_cbs)?;
    let input_count = key
        .nodes
        .iter()
        .filter(|node| node.kind == FusedElementwiseKind::Input)
        .count();
    let reader_dynamic_indices: Vec<usize> = (0..input_count).collect();

    let mut runtime_args = RuntimeArgsBuilder::new(0, vec![0], reader_dynamic_indices, Vec::new());
    for (core_index, &core) in key.cores.iter().enumerate() {
        let (offset, n_tiles) = split_tile_range(key.tile_count, core_index, key.cores.len())?;
        let offset = key
            .tile_offset
            .checked_add(offset)
            .ok_or_else(|| invalid_input("fused eltwise tile offset overflow"))?;
        let mut reader_args = vec![0; input_count];
        reader_args.push(offset);
        reader_args.push(n_tiles);
        runtime_args.add_core(core, vec![0, offset, n_tiles], reader_args, vec![n_tiles])?;
    }
    let runtime_args = runtime_args.build()?;

    let mut cbs = Vec::with_capacity(leaf_nodes.len() + intermediate_cbs.len() + 1);
    for (cb, node) in &leaf_nodes {
        cbs.push(CBConfig::new(*cb as usize, node_dtype(node)?));
    }
    for (cb, dtype) in intermediate_cbs {
        cbs.push(CBConfig::new(cb as usize, dtype));
    }
    cbs.push(CBConfig::new(16, key.output_dtype));

    let mut dst_accum_mode = matches!(
        key.output_dtype,
        DType::Float32 | DType::Int32 | DType::UInt32
    );
    for node in &key.nodes {
        if matches!(
            node_dtype(node)?,
            DType::Float32 | DType::Int32 | DType::UInt32
        ) || node.kind == FusedElementwiseKind::Select
        {
            dst_accum_mode = true;
            break;
        }
    }

    Ok(Program {
        reader_kernel: reader_source(
            &leaf_nodes,
            input_count,
            &key.input_specs,
            &key.output_shape,
        )?,
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

fn fused_leaf_nodes<'a>(
    nodes: &'a [FusedElementwiseNode],
    node_cbs: &[Option<u32>],
) -> io::Result<Vec<(u32, &'a FusedElementwiseNode)>> {
    nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| {
            matches!(
                node.kind,
                FusedElementwiseKind::Input | FusedElementwiseKind::Constant
            )
        })
        .map(|(index, node)| Ok((cb_for_node(node_cbs, index)?, node)))
        .collect()
}

fn reader_source(
    leaf_nodes: &[(u32, &FusedElementwiseNode)],
    input_count: usize,
    input_specs: &[FusedInputSpec],
    output_shape: &[u32],
) -> io::Result<String> {
    let mut arg_loads = String::new();
    let mut addr_gens = String::new();
    let mut reserves = String::new();
    let mut geometry = String::new();
    let mut reads = String::new();
    let mut broadcasts = String::new();
    let mut pushes = String::new();
    let has_input_broadcast = input_specs
        .iter()
        .any(|spec| !spec.broadcast_dimensions.is_empty());
    let (output_tile_rows, output_tiles_per_row) = if has_input_broadcast {
        tile_grid(output_shape, "fused eltwise output")?
    } else {
        (0, 0)
    };
    for index in 0..input_count {
        writeln!(
            arg_loads,
            "  uint32_t input_addr_{index} = get_arg_val<uint32_t>({index});"
        )
        .unwrap();
    }

    let mut input_arg_index = 0usize;
    for (leaf_index, (cb, node)) in leaf_nodes.iter().enumerate() {
        writeln!(
            addr_gens,
            "  constexpr uint32_t cb_leaf_{leaf_index} = tt::CBIndex::c_{cb};"
        )
        .unwrap();
        writeln!(reserves, "    cb_reserve_back(cb_leaf_{leaf_index}, 1);").unwrap();
        match node.kind {
            FusedElementwiseKind::Input => {
                writeln!(
                    addr_gens,
                    "  const InterleavedAddrGenFast<true> input_{input_arg_index} = {{"
                )
                .unwrap();
                writeln!(
                    addr_gens,
                    "    .bank_base_address = input_addr_{input_arg_index}, .page_size = get_tile_size(cb_leaf_{leaf_index}), .data_format = get_dataformat(cb_leaf_{leaf_index}),"
                )
                .unwrap();
                writeln!(addr_gens, "  }};").unwrap();
                let input_spec = input_specs.get(input_arg_index).ok_or_else(|| {
                    invalid_input(format!(
                        "missing fused input spec for input {input_arg_index}"
                    ))
                })?;
                let broadcast_kind = fused_broadcast_kind(
                    &input_spec.shape,
                    output_shape,
                    &input_spec.broadcast_dimensions,
                );
                if let Some(kind) = broadcast_kind {
                    let (input_tile_rows, input_tiles_per_row) = tile_grid(
                        &input_spec.shape,
                        &format!("fused eltwise input {input_arg_index}"),
                    )?;
                    writeln!(
                        geometry,
                        "    uint32_t input_tile_leaf_{leaf_index} = {};",
                        broadcast_tile_expr(
                            kind,
                            input_arg_index,
                            input_tile_rows,
                            input_tiles_per_row,
                            output_tile_rows,
                            output_tiles_per_row,
                        )
                    )
                    .unwrap();
                    writeln!(
                        reads,
                        "    noc_async_read_tile(input_tile_leaf_{leaf_index}, input_{input_arg_index}, get_write_ptr(cb_leaf_{leaf_index}));"
                    )
                    .unwrap();
                    let element = element_type(node_dtype(node)?);
                    match kind {
                        FusedBroadcastKind::DirectFullTile => {}
                        FusedBroadcastKind::ColumnFill => {
                            writeln!(
                                broadcasts,
                                "    fill_broadcast_columns<{element}>(cb_leaf_{leaf_index}, fused_row_count, fused_col_count);"
                            )
                            .unwrap();
                        }
                        FusedBroadcastKind::RowFill => {
                            writeln!(
                                broadcasts,
                                "    fill_broadcast_rows<{element}>(cb_leaf_{leaf_index}, fused_row_count, fused_col_count);"
                            )
                            .unwrap();
                        }
                    }
                } else {
                    let tile_id = if node.single_tile_broadcast {
                        "0"
                    } else {
                        "output_tile_id"
                    };
                    writeln!(
                        reads,
                        "    noc_async_read_tile({tile_id}, input_{input_arg_index}, get_write_ptr(cb_leaf_{leaf_index}));"
                    )
                    .unwrap();
                }
                if node.single_tile_broadcast {
                    let bytes = element_bytes(node_dtype(node)?);
                    writeln!(
                        broadcasts,
                        "    replicate_first_element(cb_leaf_{leaf_index}, {bytes});"
                    )
                    .unwrap();
                }
                input_arg_index += 1;
            }
            FusedElementwiseKind::Constant => {
                let bytes = element_bytes(node_dtype(node)?);
                writeln!(
                    broadcasts,
                    "    fill_constant_tile(cb_leaf_{leaf_index}, {}u, {bytes});",
                    node.packed_value
                )
                .unwrap();
            }
            _ => unreachable!("fused reader leaves are limited to inputs and constants"),
        }
        writeln!(pushes, "    cb_push_back(cb_leaf_{leaf_index}, 1);").unwrap();
    }

    let broadcast_geometry = if has_input_broadcast {
        format!(
            "    uint32_t fused_output_matrix_tiles = {output_tile_rows}u * {output_tiles_per_row}u;\n\
             uint32_t fused_output_batch = output_tile_id / fused_output_matrix_tiles;\n\
             uint32_t fused_output_matrix_tile = output_tile_id % fused_output_matrix_tiles;\n\
             uint32_t fused_output_tile_row = fused_output_matrix_tile / {output_tiles_per_row}u;\n\
             uint32_t fused_output_tile_col = fused_output_matrix_tile % {output_tiles_per_row}u;\n\
             uint32_t fused_row_count = tile_extent({}, fused_output_tile_row * 32u, 32u);\n\
             uint32_t fused_col_count = tile_extent({}, fused_output_tile_col * 32u, 32u);\n",
            output_shape[output_shape.len() - 2],
            output_shape[output_shape.len() - 1]
        )
    } else {
        String::new()
    };
    let output_shape_source = cpp_u32_array(output_shape);
    let mut input_shape_sources = String::new();
    for (index, spec) in input_specs.iter().enumerate() {
        if spec.broadcast_dimensions.is_empty() {
            continue;
        }
        writeln!(
            input_shape_sources,
            "constexpr uint32_t input_rank_{index} = {}u;",
            spec.shape.len()
        )
        .unwrap();
        writeln!(
            input_shape_sources,
            "constexpr uint32_t input_shape_{index}[input_rank_{index}] = {};",
            cpp_u32_array(&spec.shape)
        )
        .unwrap();
        writeln!(
            input_shape_sources,
            "constexpr uint32_t broadcast_dims_{index}[input_rank_{index}] = {};",
            cpp_u32_array(&spec.broadcast_dimensions)
        )
        .unwrap();
    }

    Ok(format!(
        "#include <cstdint>\n\
         \n\
         namespace {{\n\
         constexpr uint32_t output_rank = {}u;\n\
         constexpr uint32_t output_coord_count = output_rank == 0u ? 1u : output_rank;\n\
         constexpr uint32_t output_shape[output_coord_count] = {output_shape_source};\n\
         {input_shape_sources}\
         uint32_t repeated_word(uint32_t packed_value, uint32_t element_bytes) {{\n\
           if (element_bytes == 1) {{\n\
             uint32_t byte = packed_value & 0xffu;\n\
             return byte | (byte << 8) | (byte << 16) | (byte << 24);\n\
           }}\n\
           if (element_bytes == 2) {{\n\
             uint32_t half = packed_value & 0xffffu;\n\
             return half | (half << 16);\n\
           }}\n\
           return packed_value;\n\
         }}\n\
         void fill_constant_tile(uint32_t cb, uint32_t packed_value, uint32_t element_bytes) {{\n\
           uint32_t l1_addr = get_write_ptr(cb);\n\
           volatile tt_l1_ptr uint32_t *ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(l1_addr);\n\
           uint32_t word = repeated_word(packed_value, element_bytes);\n\
           uint32_t words = get_tile_size(cb) / sizeof(uint32_t);\n\
           for (uint32_t i = 0; i < words; ++i) {{\n\
             ptr[i] = word;\n\
           }}\n\
         }}\n\
         void replicate_first_element(uint32_t cb, uint32_t element_bytes) {{\n\
           uint32_t l1_addr = get_write_ptr(cb);\n\
           volatile tt_l1_ptr uint32_t *ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(l1_addr);\n\
           uint32_t packed_value = repeated_word(ptr[0], element_bytes);\n\
           uint32_t words = get_tile_size(cb) / sizeof(uint32_t);\n\
           for (uint32_t i = 0; i < words; ++i) {{\n\
             ptr[i] = packed_value;\n\
           }}\n\
         }}\n\
         uint32_t tile_element_index(uint32_t row, uint32_t col) {{\n\
           uint32_t face_row = row / 16u;\n\
           uint32_t face_col = col / 16u;\n\
           uint32_t row_in_face = row % 16u;\n\
           uint32_t col_in_face = col % 16u;\n\
           return ((face_row * 2u + face_col) * 16u * 16u) + row_in_face * 16u + col_in_face;\n\
         }}\n\
         uint32_t tile_extent(uint32_t logical_dim, uint32_t base, uint32_t tile_dim) {{\n\
           if (base >= logical_dim) {{\n\
             return 0;\n\
           }}\n\
           uint32_t remaining = logical_dim - base;\n\
           return remaining < tile_dim ? remaining : tile_dim;\n\
         }}\n\
         void zero_tile(uint32_t cb) {{\n\
           volatile tt_l1_ptr uint32_t *ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_write_ptr(cb));\n\
           uint32_t words = get_tile_size(cb) / sizeof(uint32_t);\n\
           for (uint32_t i = 0; i < words; ++i) {{\n\
             ptr[i] = 0;\n\
           }}\n\
         }}\n\
         uint32_t output_batch_coord(uint32_t output_batch, uint32_t dim) {{\n\
           if (output_rank < 3u || dim >= output_rank - 2u) {{\n\
             return 0;\n\
           }}\n\
           for (uint32_t rev = 0; rev < output_rank - 2u; ++rev) {{\n\
             uint32_t current = output_rank - 3u - rev;\n\
             uint32_t coord = output_batch % output_shape[current];\n\
             if (current == dim) {{\n\
               return coord;\n\
             }}\n\
             output_batch /= output_shape[current];\n\
           }}\n\
           return 0;\n\
         }}\n\
         uint32_t direct_broadcast_tile(uint32_t output_batch, uint32_t output_tile_row, uint32_t output_tile_col, const uint32_t *input_shape, uint32_t input_rank, const uint32_t *broadcast_dims, uint32_t input_tile_rows, uint32_t input_tiles_per_row) {{\n\
           uint32_t input_batch = 0;\n\
           for (uint32_t dim = 0; dim < input_rank - 2u; ++dim) {{\n\
             uint32_t coord = input_shape[dim] == 1u ? 0u : output_batch_coord(output_batch, broadcast_dims[dim]);\n\
             input_batch = input_batch * input_shape[dim] + coord;\n\
           }}\n\
           return (input_batch * input_tile_rows + output_tile_row) * input_tiles_per_row + output_tile_col;\n\
         }}\n\
         template <typename Element>\n\
         void fill_broadcast_columns(uint32_t cb, uint32_t row_count, uint32_t col_count) {{\n\
           volatile tt_l1_ptr Element *ptr = reinterpret_cast<volatile tt_l1_ptr Element *>(get_write_ptr(cb));\n\
           Element values[32];\n\
           for (uint32_t row = 0; row < row_count; ++row) {{\n\
             values[row] = ptr[tile_element_index(row, 0u)];\n\
           }}\n\
           zero_tile(cb);\n\
           for (uint32_t row = 0; row < row_count; ++row) {{\n\
             Element value = values[row];\n\
             for (uint32_t col = 0; col < col_count; ++col) {{\n\
               ptr[tile_element_index(row, col)] = value;\n\
             }}\n\
           }}\n\
         }}\n\
         template <typename Element>\n\
         void fill_broadcast_rows(uint32_t cb, uint32_t row_count, uint32_t col_count) {{\n\
           volatile tt_l1_ptr Element *ptr = reinterpret_cast<volatile tt_l1_ptr Element *>(get_write_ptr(cb));\n\
           Element values[32];\n\
           for (uint32_t col = 0; col < col_count; ++col) {{\n\
             values[col] = ptr[tile_element_index(0u, col)];\n\
           }}\n\
           zero_tile(cb);\n\
           for (uint32_t row = 0; row < row_count; ++row) {{\n\
             for (uint32_t col = 0; col < col_count; ++col) {{\n\
               ptr[tile_element_index(row, col)] = values[col];\n\
             }}\n\
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
             uint32_t output_tile_id = offset + i;\n\
         {broadcast_geometry}\
         {geometry}\
         {reserves}\
         {reads}\
             noc_async_read_barrier();\n\
         {broadcasts}\
        {pushes}\
           }}\n\
         }}\n",
        output_shape.len(),
        input_count + 1
    ))
}

fn broadcast_tile_expr(
    kind: FusedBroadcastKind,
    input_index: usize,
    input_tile_rows: u32,
    input_tiles_per_row: u32,
    _output_tile_rows: u32,
    _output_tiles_per_row: u32,
) -> String {
    match kind {
        FusedBroadcastKind::DirectFullTile => format!(
            "direct_broadcast_tile(fused_output_batch, fused_output_tile_row, fused_output_tile_col, input_shape_{input_index}, input_rank_{input_index}, broadcast_dims_{input_index}, {input_tile_rows}u, {input_tiles_per_row}u)"
        ),
        FusedBroadcastKind::ColumnFill => format!(
            "(fused_output_batch * {input_tile_rows}u + fused_output_tile_row) * {input_tiles_per_row}u"
        ),
        FusedBroadcastKind::RowFill => format!(
            "(fused_output_batch * {input_tile_rows}u) * {input_tiles_per_row}u + fused_output_tile_col"
        ),
    }
}

fn tile_grid(shape: &[u32], name: &str) -> io::Result<(u32, u32)> {
    let shape = usize_shape(shape)?;
    let allocation_shape = tiled_allocation_shape(&shape)?;
    let rank = allocation_shape.len();
    Ok((
        u32_arg(
            allocation_shape[rank - 2] / TILE_R,
            &format!("{name} tile rows"),
        )?,
        u32_arg(
            allocation_shape[rank - 1] / TILE_C,
            &format!("{name} tiles per row"),
        )?,
    ))
}

fn cpp_u32_array(values: &[u32]) -> String {
    if values.is_empty() {
        return "{1u}".to_owned();
    }
    let values = values
        .iter()
        .map(|value| format!("{value}u"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{{{values}}}")
}

fn element_type(dtype: DType) -> &'static str {
    match dtype {
        DType::Float32 | DType::Int32 | DType::UInt32 => "uint32_t",
        DType::Float16 | DType::Float16B | DType::UInt16 => "uint16_t",
        DType::Int8 | DType::UInt8 => "uint8_t",
    }
}

fn compute_source(key: &FusedEltwiseProgramKey) -> io::Result<String> {
    let steps = compute_steps(&key.nodes)?;
    Ok(COMPUTE
        .replace("FUSED_HEADERS", &steps.features.headers_source())
        .replace("FUSED_HELPERS", &steps.features.helpers_source())
        .replace("FUSED_STEPS", &steps.body))
}

#[derive(Default)]
struct ComputeSourceFeatures {
    headers: Vec<&'static str>,
    add_input_helper: bool,
    subtract_input_helper: bool,
    multiply_input_helper: bool,
    compare_helpers: bool,
    select_helper: bool,
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

    fn add_select_helper(&mut self) {
        self.select_helper = true;
        self.add_header(HEADER_BINARY_SFPU);
        self.add_header(HEADER_ADD_INT);
        self.add_header(HEADER_SFPU_SPLIT);
    }

    fn add_data_format_binary_helper(&mut self, op: FusedElementwiseKind) {
        self.add_header(HEADER_BINARY_SFPU);
        match op {
            FusedElementwiseKind::Add => {
                self.add_input_helper = true;
                self.add_header(HEADER_ADD_INT);
            }
            FusedElementwiseKind::Subtract => {
                self.subtract_input_helper = true;
                self.add_header(HEADER_SUB_INT);
            }
            FusedElementwiseKind::Multiply => {
                self.multiply_input_helper = true;
                self.add_header(HEADER_MUL_INT);
                self.add_header(HEADER_MUL_INT32);
            }
            _ => {}
        }
    }

    fn headers_source(&self) -> String {
        let mut source = String::new();
        for header in &self.headers {
            writeln!(source, "#include \"{header}\"").unwrap();
        }
        source
    }

    fn helpers_source(&self) -> String {
        let mut helpers = String::new();
        if self.add_input_helper || self.subtract_input_helper || self.multiply_input_helper {
            helpers.push_str(HELPER_BINARY_INPUT_DATA_FORMAT);
        }
        if self.add_input_helper {
            helpers.push_str(HELPER_ADD_INPUT);
        }
        if self.subtract_input_helper {
            helpers.push_str(HELPER_SUBTRACT_INPUT);
        }
        if self.multiply_input_helper {
            helpers.push_str(HELPER_MULTIPLY_INPUT);
        }
        if self.compare_helpers {
            helpers.push_str(HELPER_COMPARE);
        }
        if self.select_helper {
            helpers.push_str(HELPER_SELECT);
        }
        helpers
    }
}

struct ComputeSteps {
    body: String,
    features: ComputeSourceFeatures,
}

struct ComputeEmitContext<'a> {
    node_cbs: &'a [Option<u32>],
    cb_dtypes: &'a [Option<DType>; 32],
    remaining_uses: &'a mut [u32],
    srca_cb: u32,
    pack_cb: u32,
    body: String,
    features: ComputeSourceFeatures,
}

impl ComputeEmitContext<'_> {
    fn cb_for_node(&self, node: usize) -> io::Result<u32> {
        cb_for_node(self.node_cbs, node)
    }

    fn pack_and_pop(&mut self, output_cb: u32, input_nodes: &[usize]) -> io::Result<()> {
        let old_pack_dtype = cb_dtype(self.cb_dtypes, self.pack_cb)?;
        let output_dtype = cb_dtype(self.cb_dtypes, output_cb)?;
        append_pack_and_pop(
            &mut self.body,
            self.pack_cb,
            old_pack_dtype != output_dtype,
            output_cb,
            input_nodes,
            self.node_cbs,
            self.remaining_uses,
        )?;
        self.pack_cb = output_cb;
        Ok(())
    }

    fn copy_to_dst(&mut self, cb: u32, dst: u32) {
        writeln!(
            self.body,
            "    copy_tile_to_dst_init_short_with_dt(tt::CBIndex::c_{}, tt::CBIndex::c_{cb});",
            self.srca_cb
        )
        .unwrap();
        writeln!(self.body, "    copy_tile(tt::CBIndex::c_{cb}, 0, {dst});").unwrap();
        self.srca_cb = cb;
    }
}

#[derive(Clone, Copy)]
enum Lowering {
    Leaf,
    Unary(UnarySpec),
    ScalarBinary(ScalarBinarySpec),
    Binary(BinarySpec),
    Compare(CompareSpec),
    Select(SelectSpec),
}

#[derive(Clone, Copy)]
struct UnarySpec {
    input: usize,
    op: UnaryLowering,
}

#[derive(Clone, Copy)]
enum UnaryLowering {
    Tile(UnaryCompute),
    Convert { from: u32, to: u32 },
}

#[derive(Clone, Copy)]
struct ScalarBinarySpec {
    value_node: usize,
    init: &'static str,
    tile: &'static str,
    scalar: u32,
    feature: ScalarBinaryFeature,
}

#[derive(Clone, Copy)]
enum ScalarBinaryFeature {
    Scalar(ScalarCompute),
    UnaryCompare,
}

#[derive(Clone, Copy)]
struct BinarySpec {
    lhs: usize,
    rhs: usize,
    op: BinaryLowering,
}

#[derive(Clone, Copy)]
enum BinaryLowering {
    DataFormatHelper {
        helper: &'static str,
        op: FusedElementwiseKind,
    },
    Tile(BinaryCompute),
}

#[derive(Clone, Copy)]
struct CompareSpec {
    lhs: usize,
    rhs: usize,
    direction: CompareDirection,
    int32_input: bool,
}

#[derive(Clone, Copy)]
struct SelectSpec {
    pred: usize,
    on_true: usize,
    on_false: usize,
    int32_value: bool,
}

fn compute_steps(nodes: &[FusedElementwiseNode]) -> io::Result<ComputeSteps> {
    let mut remaining_uses = vec![0u32; nodes.len()];
    for node in nodes {
        for &input_node in &node.input_nodes {
            let index = input_node as usize;
            if index >= nodes.len() {
                return Err(invalid_input(format!(
                    "node id out of bounds: {input_node}"
                )));
            }
            remaining_uses[index] += 1;
        }
    }

    let (node_cbs, _) = cb_plan(nodes)?;
    let mut cb_dtypes = [None; 32];
    for (node, cb) in nodes.iter().zip(node_cbs.iter().copied()) {
        if let Some(cb) = cb {
            let index = cb as usize;
            if index >= cb_dtypes.len() {
                return Err(invalid_input(format!("CB id out of bounds: {cb}")));
            }
            cb_dtypes[index] = Some(node_dtype(node)?);
        }
    }
    let mut ctx = ComputeEmitContext {
        node_cbs: &node_cbs,
        cb_dtypes: &cb_dtypes,
        remaining_uses: &mut remaining_uses,
        srca_cb: 0,
        pack_cb: 16,
        body: String::new(),
        features: ComputeSourceFeatures::default(),
    };

    for (index, node) in nodes.iter().enumerate() {
        match lowering_for(node, nodes)? {
            Lowering::Leaf => {}
            Lowering::Unary(spec) => emit_unary(&mut ctx, index, spec)?,
            Lowering::ScalarBinary(spec) => emit_scalar_binary(&mut ctx, index, spec)?,
            Lowering::Binary(spec) => emit_binary(&mut ctx, index, spec)?,
            Lowering::Compare(spec) => emit_compare(&mut ctx, index, spec)?,
            Lowering::Select(spec) => emit_select(&mut ctx, index, spec)?,
        }
    }

    Ok(ComputeSteps {
        body: ctx.body,
        features: ctx.features,
    })
}

fn lowering_for(
    node: &FusedElementwiseNode,
    nodes: &[FusedElementwiseNode],
) -> io::Result<Lowering> {
    match node.kind.arity() {
        0 => Ok(Lowering::Leaf),
        1 => {
            let input = node.input_nodes[0] as usize;
            let op = if let Some(unary) = node.kind.unary_compute() {
                UnaryLowering::Tile(unary)
            } else {
                debug_assert_eq!(node.kind, FusedElementwiseKind::Convert);
                UnaryLowering::Convert {
                    from: node_dtype(&nodes[input])? as u32,
                    to: node_dtype(node)? as u32,
                }
            };
            Ok(Lowering::Unary(UnarySpec { input, op }))
        }
        2 => {
            let lhs = node.input_nodes[0] as usize;
            let rhs = node.input_nodes[1] as usize;
            if let FusedElementwiseKind::Compare(direction) = node.kind {
                if let Some((value_node, scalar, scalar_direction)) =
                    scalar_compare_op(nodes, lhs, rhs, direction)?
                {
                    let int32_input = node_dtype(&nodes[value_node])? == DType::Int32;
                    return Ok(Lowering::ScalarBinary(ScalarBinarySpec {
                        value_node,
                        init: scalar_direction.unary_init(),
                        tile: scalar_direction.unary_tile(int32_input),
                        scalar,
                        feature: ScalarBinaryFeature::UnaryCompare,
                    }));
                }
            }
            if let Some(scalar_op) = node.kind.scalar_compute(
                node_dtype(&nodes[lhs])?,
                node_dtype(&nodes[rhs])?,
                constant_scalar_bits(nodes, lhs)?,
                constant_scalar_bits(nodes, rhs)?,
            ) {
                let value_node = match scalar_op.operand {
                    ScalarOperand::Lhs => lhs,
                    ScalarOperand::Rhs => rhs,
                };
                return Ok(Lowering::ScalarBinary(ScalarBinarySpec {
                    value_node,
                    init: scalar_op.init,
                    tile: scalar_op.tile,
                    scalar: scalar_op.scalar,
                    feature: ScalarBinaryFeature::Scalar(scalar_op),
                }));
            }
            if let FusedElementwiseKind::Compare(direction) = node.kind {
                return Ok(Lowering::Compare(CompareSpec {
                    lhs,
                    rhs,
                    direction,
                    int32_input: node_dtype(&nodes[lhs])? == DType::Int32,
                }));
            }
            if let Some(helper) = node.kind.data_format_binary_helper() {
                return Ok(Lowering::Binary(BinarySpec {
                    lhs,
                    rhs,
                    op: BinaryLowering::DataFormatHelper {
                        helper,
                        op: node.kind,
                    },
                }));
            }
            let binary = node.kind.binary_compute().ok_or_else(|| {
                invalid_input(format!("missing binary lowering for {:?}", node.kind))
            })?;
            Ok(Lowering::Binary(BinarySpec {
                lhs,
                rhs,
                op: BinaryLowering::Tile(binary),
            }))
        }
        3 => {
            if node.kind != FusedElementwiseKind::Select {
                return Err(invalid_input(format!(
                    "missing ternary lowering for {:?}",
                    node.kind
                )));
            }
            let pred = node.input_nodes[0] as usize;
            let on_true = node.input_nodes[1] as usize;
            let on_false = node.input_nodes[2] as usize;
            Ok(Lowering::Select(SelectSpec {
                pred,
                on_true,
                on_false,
                int32_value: node_dtype(&nodes[on_true])? == DType::Int32,
            }))
        }
        _ => unreachable!("fused eltwise op arity is limited to 0, 1, 2, or 3"),
    }
}

fn emit_unary(ctx: &mut ComputeEmitContext<'_>, index: usize, spec: UnarySpec) -> io::Result<()> {
    let input_cb = ctx.cb_for_node(spec.input)?;
    let output_cb = ctx.cb_for_node(index)?;
    append_waits(&mut ctx.body, &[input_cb]);
    let init = match spec.op {
        UnaryLowering::Tile(unary) => unary.init,
        UnaryLowering::Convert { .. } => "",
    };
    if !init.is_empty() {
        writeln!(ctx.body, "    {init}").unwrap();
    }
    writeln!(
        ctx.body,
        "    cb_reserve_back(tt::CBIndex::c_{output_cb}, 1);"
    )
    .unwrap();
    writeln!(ctx.body, "    tile_regs_acquire();").unwrap();
    ctx.copy_to_dst(input_cb, 0);
    match spec.op {
        UnaryLowering::Tile(unary) => {
            ctx.features.add_unary(unary);
            writeln!(ctx.body, "    {}", unary.tile).unwrap();
        }
        UnaryLowering::Convert { from, to } => {
            ctx.features.add_typecast();
            writeln!(ctx.body, "    typecast_tile_init<{from}, {to}>();").unwrap();
            writeln!(ctx.body, "    typecast_tile<{from}, {to}>(0);").unwrap();
        }
    }
    ctx.pack_and_pop(output_cb, &[spec.input])
}

fn emit_scalar_binary(
    ctx: &mut ComputeEmitContext<'_>,
    index: usize,
    spec: ScalarBinarySpec,
) -> io::Result<()> {
    let value_cb = ctx.cb_for_node(spec.value_node)?;
    let output_cb = ctx.cb_for_node(index)?;
    append_waits(&mut ctx.body, &[value_cb]);
    match spec.feature {
        ScalarBinaryFeature::Scalar(scalar) => ctx.features.add_scalar(scalar),
        ScalarBinaryFeature::UnaryCompare => ctx.features.add_unary_compare(),
    }
    writeln!(ctx.body, "    {}();", spec.init).unwrap();
    writeln!(
        ctx.body,
        "    cb_reserve_back(tt::CBIndex::c_{output_cb}, 1);"
    )
    .unwrap();
    writeln!(ctx.body, "    tile_regs_acquire();").unwrap();
    ctx.copy_to_dst(value_cb, 0);
    writeln!(ctx.body, "    {}(0, {});", spec.tile, spec.scalar).unwrap();
    ctx.pack_and_pop(output_cb, &[spec.value_node])
}

fn emit_binary(ctx: &mut ComputeEmitContext<'_>, index: usize, spec: BinarySpec) -> io::Result<()> {
    let lhs_cb = ctx.cb_for_node(spec.lhs)?;
    let rhs_cb = ctx.cb_for_node(spec.rhs)?;
    let output_cb = ctx.cb_for_node(index)?;
    append_waits(&mut ctx.body, &[lhs_cb, rhs_cb]);
    match spec.op {
        BinaryLowering::DataFormatHelper { helper, op } => {
            ctx.features.add_data_format_binary_helper(op);
            writeln!(
                ctx.body,
                "    constexpr DataFormat input_format_{index} = binary_input_data_format(tt::CBIndex::c_{lhs_cb}, tt::CBIndex::c_{output_cb});"
            )
            .unwrap();
            writeln!(ctx.body, "    {helper}_init<input_format_{index}>();").unwrap();
            append_binary_tile_setup(ctx, output_cb, lhs_cb, rhs_cb);
            writeln!(
                ctx.body,
                "    {helper}_tile<input_format_{index}>(0, 1, 0);"
            )
            .unwrap();
        }
        BinaryLowering::Tile(binary) => {
            ctx.features.add_binary(binary);
            writeln!(ctx.body, "    {}();", binary.init).unwrap();
            append_binary_tile_setup(ctx, output_cb, lhs_cb, rhs_cb);
            writeln!(ctx.body, "    {}(0, 1, 0);", binary.tile).unwrap();
        }
    }
    ctx.pack_and_pop(output_cb, &[spec.lhs, spec.rhs])
}

fn emit_compare(
    ctx: &mut ComputeEmitContext<'_>,
    index: usize,
    spec: CompareSpec,
) -> io::Result<()> {
    let lhs_cb = ctx.cb_for_node(spec.lhs)?;
    let rhs_cb = ctx.cb_for_node(spec.rhs)?;
    let output_cb = ctx.cb_for_node(index)?;
    let int32_input = spec.int32_input;
    let direction = spec.direction.variant();
    append_waits(&mut ctx.body, &[lhs_cb, rhs_cb]);
    ctx.features.add_compare_helpers();
    writeln!(ctx.body, "    compare_sub_init<{int32_input}>();").unwrap();
    writeln!(
        ctx.body,
        "    compare_zero_init(CompareDirection::{direction});"
    )
    .unwrap();
    append_binary_tile_setup(ctx, output_cb, lhs_cb, rhs_cb);
    writeln!(ctx.body, "    compare_sub_tile<{int32_input}>(0, 1, 0);").unwrap();
    writeln!(
        ctx.body,
        "    compare_zero_tile<{int32_input}>(CompareDirection::{direction}, 0);"
    )
    .unwrap();
    ctx.pack_and_pop(output_cb, &[spec.lhs, spec.rhs])
}

fn emit_select(ctx: &mut ComputeEmitContext<'_>, index: usize, spec: SelectSpec) -> io::Result<()> {
    let pred_cb = ctx.cb_for_node(spec.pred)?;
    let true_cb = ctx.cb_for_node(spec.on_true)?;
    let false_cb = ctx.cb_for_node(spec.on_false)?;
    let output_cb = ctx.cb_for_node(index)?;
    let int32_value = spec.int32_value;
    append_waits(&mut ctx.body, &[pred_cb, true_cb, false_cb]);
    ctx.features.add_select_helper();
    writeln!(
        ctx.body,
        "    constexpr DataFormat select_format_{index} = fused_select_value_format(tt::CBIndex::c_{true_cb}, tt::CBIndex::c_{output_cb});"
    )
    .unwrap();
    writeln!(
        ctx.body,
        "    fused_select_add_init<select_format_{index}>();"
    )
    .unwrap();
    writeln!(
        ctx.body,
        "    cb_reserve_back(tt::CBIndex::c_{output_cb}, 1);"
    )
    .unwrap();
    writeln!(ctx.body, "    tile_regs_acquire();").unwrap();
    writeln!(
        ctx.body,
        "    reconfig_data_format_srca<true>(tt::CBIndex::c_{pred_cb});"
    )
    .unwrap();
    writeln!(
        ctx.body,
        "    copy_tile_to_dst_init_short(tt::CBIndex::c_{pred_cb});"
    )
    .unwrap();
    writeln!(ctx.body, "    copy_tile(tt::CBIndex::c_{pred_cb}, 0, 0);").unwrap();
    writeln!(
        ctx.body,
        "    reconfig_data_format_srca<true>(tt::CBIndex::c_{true_cb});"
    )
    .unwrap();
    writeln!(ctx.body, "    copy_tile_init(tt::CBIndex::c_{true_cb});").unwrap();
    writeln!(ctx.body, "    copy_tile(tt::CBIndex::c_{true_cb}, 0, 1);").unwrap();
    writeln!(
        ctx.body,
        "    MATH(_llk_math_eltwise_binary_sfpu_params_<false>(fused_select_gate_value<true, {int32_value}>, 0, 1, 1, VectorMode::RC);)"
    )
    .unwrap();
    writeln!(ctx.body, "    copy_tile_init(tt::CBIndex::c_{false_cb});").unwrap();
    writeln!(ctx.body, "    copy_tile(tt::CBIndex::c_{false_cb}, 0, 2);").unwrap();
    writeln!(
        ctx.body,
        "    MATH(_llk_math_eltwise_binary_sfpu_params_<false>(fused_select_gate_value<false, {int32_value}>, 0, 2, 2, VectorMode::RC);)"
    )
    .unwrap();
    writeln!(
        ctx.body,
        "    fused_select_add_tile<select_format_{index}>(1, 2, 0);"
    )
    .unwrap();
    ctx.srca_cb = false_cb;
    ctx.pack_and_pop(output_cb, &[spec.pred, spec.on_true, spec.on_false])
}

fn append_binary_tile_setup(
    ctx: &mut ComputeEmitContext<'_>,
    output_cb: u32,
    lhs_cb: u32,
    rhs_cb: u32,
) {
    writeln!(
        ctx.body,
        "    cb_reserve_back(tt::CBIndex::c_{output_cb}, 1);"
    )
    .unwrap();
    writeln!(ctx.body, "    tile_regs_acquire();").unwrap();
    ctx.copy_to_dst(lhs_cb, 0);
    ctx.copy_to_dst(rhs_cb, 1);
}

fn cb_plan(nodes: &[FusedElementwiseNode]) -> io::Result<(Vec<Option<u32>>, Vec<(u32, DType)>)> {
    let mut node_cbs = vec![None; nodes.len()];
    let mut leaf_count = 0u32;
    for (index, node) in nodes.iter().enumerate() {
        if matches!(
            node.kind,
            FusedElementwiseKind::Input | FusedElementwiseKind::Constant
        ) {
            if leaf_count >= 16 {
                return Err(invalid_input("fused eltwise needs too many leaf CBs"));
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
        if matches!(
            node.kind,
            FusedElementwiseKind::Input | FusedElementwiseKind::Constant
        ) {
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
            intermediate_cbs.push((next_cb, node_dtype(node)?));
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

fn cb_dtype(cb_dtypes: &[Option<DType>; 32], cb: u32) -> io::Result<DType> {
    let index = cb as usize;
    cb_dtypes
        .get(index)
        .and_then(|dtype| *dtype)
        .ok_or_else(|| invalid_input(format!("CB {cb} does not have a dtype")))
}

fn append_waits(body: &mut String, cbs: &[u32]) {
    for (index, &cb) in cbs.iter().enumerate() {
        if cbs[..index].contains(&cb) {
            continue;
        }
        writeln!(body, "    cb_wait_front(tt::CBIndex::c_{cb}, 1);").unwrap();
    }
}

fn append_pack_and_pop(
    body: &mut String,
    previous_output_cb: u32,
    reconfigure_pack: bool,
    output_cb: u32,
    input_nodes: &[usize],
    node_cbs: &[Option<u32>],
    remaining_uses: &mut [u32],
) -> io::Result<()> {
    writeln!(body, "    tile_regs_commit();").unwrap();
    writeln!(body, "    tile_regs_wait();").unwrap();
    if reconfigure_pack {
        writeln!(
            body,
            "    pack_reconfig_data_format(tt::CBIndex::c_{previous_output_cb}, tt::CBIndex::c_{output_cb});"
        )
        .unwrap();
    }
    writeln!(body, "    pack_tile(0, tt::CBIndex::c_{output_cb});").unwrap();
    writeln!(body, "    tile_regs_release();").unwrap();

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
                writeln!(body, "    cb_pop_front(tt::CBIndex::c_{cb}, 1);").unwrap();
            }
        }
    }
    writeln!(body, "    cb_push_back(tt::CBIndex::c_{output_cb}, 1);").unwrap();
    Ok(())
}

fn scalar_compare_op(
    nodes: &[FusedElementwiseNode],
    lhs: usize,
    rhs: usize,
    direction: CompareDirection,
) -> io::Result<Option<(usize, u32, CompareDirection)>> {
    let lhs_constant = constant_scalar_bits(nodes, lhs)?;
    let rhs_constant = constant_scalar_bits(nodes, rhs)?;
    Ok(match (lhs_constant, rhs_constant) {
        (None, Some(scalar)) => Some((lhs, scalar, direction)),
        (Some(scalar), None) => Some((rhs, scalar, direction.reversed())),
        _ => None,
    })
}

fn constant_scalar_bits(nodes: &[FusedElementwiseNode], index: usize) -> io::Result<Option<u32>> {
    let node = &nodes[index];
    if node.kind != FusedElementwiseKind::Constant {
        return Ok(None);
    }
    Ok(Some(match node_dtype(node)? {
        DType::Float32 => node.packed_value,
        DType::Float16B => (node.packed_value & 0xffff) << 16,
        DType::Float16 => f16_to_f32_bits((node.packed_value & 0xffff) as u16),
        _ => node.packed_value,
    }))
}

pub(crate) fn supports_input_broadcast(
    input_shape: &[usize],
    output_shape: &[usize],
    broadcast_dimensions: &[i64],
) -> bool {
    if !valid_broadcast_dimensions(input_shape, output_shape, broadcast_dimensions) {
        return false;
    }
    if direct_full_tile_broadcast(input_shape, output_shape, broadcast_dimensions) {
        return true;
    }
    if identity_column_fill_broadcast(input_shape, output_shape, broadcast_dimensions) {
        return true;
    }
    if identity_row_fill_broadcast(input_shape, output_shape, broadcast_dimensions) {
        return true;
    }
    false
}

fn valid_broadcast_dimensions(
    input_shape: &[usize],
    output_shape: &[usize],
    broadcast_dimensions: &[i64],
) -> bool {
    if broadcast_dimensions.len() != input_shape.len() {
        return false;
    }
    let mut previous = None;
    for (input_dim, &output_dim) in broadcast_dimensions.iter().enumerate() {
        let Ok(output_dim) = usize::try_from(output_dim) else {
            return false;
        };
        if output_dim >= output_shape.len() {
            return false;
        }
        if previous.is_some_and(|previous| output_dim <= previous) {
            return false;
        }
        previous = Some(output_dim);
        let input = input_shape[input_dim];
        let output = output_shape[output_dim];
        if input != output && input != 1 {
            return false;
        }
    }
    true
}

fn direct_full_tile_broadcast(
    input_shape: &[usize],
    output_shape: &[usize],
    broadcast_dimensions: &[i64],
) -> bool {
    let input_rank = input_shape.len();
    let output_rank = output_shape.len();
    input_rank >= 2
        && output_rank >= 2
        && usize::try_from(broadcast_dimensions[input_rank - 2]).ok() == Some(output_rank - 2)
        && usize::try_from(broadcast_dimensions[input_rank - 1]).ok() == Some(output_rank - 1)
        && input_shape[input_rank - 2] == output_shape[output_rank - 2]
        && input_shape[input_rank - 1] == output_shape[output_rank - 1]
}

fn identity_column_fill_broadcast(
    input_shape: &[usize],
    output_shape: &[usize],
    broadcast_dimensions: &[i64],
) -> bool {
    let rank = input_shape.len();
    rank >= 2
        && rank == output_shape.len()
        && broadcast_dimensions
            .iter()
            .enumerate()
            .all(|(index, &dim)| usize::try_from(dim).ok() == Some(index))
        && input_shape[..rank - 1] == output_shape[..rank - 1]
        && input_shape[rank - 1] == 1
        && output_shape[rank - 1] > 1
}

fn identity_row_fill_broadcast(
    input_shape: &[usize],
    output_shape: &[usize],
    broadcast_dimensions: &[i64],
) -> bool {
    let rank = input_shape.len();
    rank >= 2
        && rank == output_shape.len()
        && broadcast_dimensions
            .iter()
            .enumerate()
            .all(|(index, &dim)| usize::try_from(dim).ok() == Some(index))
        && input_shape[..rank - 2] == output_shape[..rank - 2]
        && input_shape[rank - 2] == 1
        && output_shape[rank - 2] > 1
        && input_shape[rank - 1] == output_shape[rank - 1]
}

#[derive(Clone, Copy)]
enum FusedBroadcastKind {
    DirectFullTile,
    ColumnFill,
    RowFill,
}

fn fused_broadcast_kind(
    input_shape: &[u32],
    output_shape: &[u32],
    dims: &[u32],
) -> Option<FusedBroadcastKind> {
    if dims.is_empty() || dims.len() != input_shape.len() {
        return None;
    }
    let input_shape_usize = input_shape
        .iter()
        .map(|&dim| dim as usize)
        .collect::<Vec<_>>();
    let output_shape_usize = output_shape
        .iter()
        .map(|&dim| dim as usize)
        .collect::<Vec<_>>();
    let dims_i64 = dims.iter().map(|&dim| i64::from(dim)).collect::<Vec<_>>();
    if !valid_broadcast_dimensions(&input_shape_usize, &output_shape_usize, &dims_i64) {
        return None;
    }
    if direct_full_tile_broadcast(&input_shape_usize, &output_shape_usize, &dims_i64) {
        Some(FusedBroadcastKind::DirectFullTile)
    } else if identity_column_fill_broadcast(&input_shape_usize, &output_shape_usize, &dims_i64) {
        Some(FusedBroadcastKind::ColumnFill)
    } else if identity_row_fill_broadcast(&input_shape_usize, &output_shape_usize, &dims_i64) {
        Some(FusedBroadcastKind::RowFill)
    } else {
        None
    }
}

fn u32_shape(shape: &[usize], name: &str) -> io::Result<Vec<u32>> {
    shape
        .iter()
        .map(|&dim| u32_arg(dim, name))
        .collect::<io::Result<Vec<_>>>()
}

fn u32_dims(dims: &[i64], name: &str) -> io::Result<Vec<u32>> {
    dims.iter()
        .map(|&dim| {
            u32::try_from(dim)
                .map_err(|_| invalid_input(format!("{name} does not fit in u32: {dim}")))
        })
        .collect()
}

fn usize_shape(shape: &[u32]) -> io::Result<Vec<usize>> {
    shape
        .iter()
        .map(|&dim| {
            usize::try_from(dim)
                .map_err(|_| invalid_input(format!("shape dim does not fit in usize: {dim}")))
        })
        .collect()
}

fn element_bytes(dtype: DType) -> usize {
    match dtype {
        DType::Float32 | DType::Int32 | DType::UInt32 => 4,
        DType::Float16 | DType::Float16B | DType::UInt16 => 2,
        DType::Int8 | DType::UInt8 => 1,
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

fn is_supported_value_dtype(dtype: DType) -> bool {
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

fn is_supported_convert_dtype(dtype: DType) -> bool {
    is_supported_value_dtype(dtype) || dtype == DType::UInt8
}

fn is_supported_select_value_dtype(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::Float16 | DType::Float16B | DType::Float32 | DType::Int32
    )
}

fn is_float_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::Float16 | DType::Float16B | DType::Float32)
}

fn validate_same_output_dtype(
    node_index: usize,
    op: FusedElementwiseKind,
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
    use crate::PJRT_Buffer_Type;

    fn element_type(dtype: DType) -> PJRT_Buffer_Type {
        match dtype {
            DType::Float32 => PJRT_Buffer_Type::PJRT_Buffer_Type_F32,
            DType::Float16 => PJRT_Buffer_Type::PJRT_Buffer_Type_F16,
            DType::Float16B => PJRT_Buffer_Type::PJRT_Buffer_Type_BF16,
            DType::Int32 => PJRT_Buffer_Type::PJRT_Buffer_Type_S32,
            DType::UInt16 => PJRT_Buffer_Type::PJRT_Buffer_Type_U16,
            DType::Int8 => PJRT_Buffer_Type::PJRT_Buffer_Type_S8,
            DType::UInt32 => PJRT_Buffer_Type::PJRT_Buffer_Type_U32,
            DType::UInt8 => PJRT_Buffer_Type::PJRT_Buffer_Type_U8,
        }
    }

    fn node(op: FusedElementwiseKind, input_nodes: Vec<u32>) -> FusedElementwiseNode {
        FusedElementwiseNode {
            kind: op,
            input_nodes,
            input_index: 0,
            packed_value: 0,
            element_type: element_type(DType::Float16B),
            single_tile_broadcast: false,
            broadcast_dimensions: Vec::new(),
        }
    }

    fn program_key(nodes: Vec<FusedElementwiseNode>) -> FusedEltwiseProgramKey {
        FusedEltwiseProgramKey {
            cores: Vec::new(),
            tile_offset: 0,
            tile_count: 1,
            output_shape: vec![32, 32],
            output_dtype: node_dtype(nodes.last().expect("test nodes must not be empty"))
                .expect("test node dtype must be valid"),
            input_specs: nodes
                .iter()
                .filter(|node| node.kind == FusedElementwiseKind::Input)
                .map(|_| FusedInputSpec {
                    shape: vec![32, 32],
                    broadcast_dimensions: Vec::new(),
                })
                .collect(),
            nodes,
        }
    }

    #[test]
    fn compute_steps_handles_constant_left_divide_as_rdiv() {
        let mut constant = node(FusedElementwiseKind::Constant, Vec::new());
        constant.packed_value = 0x3f80_3f80;

        let nodes = vec![
            node(FusedElementwiseKind::Input, Vec::new()),
            constant,
            node(FusedElementwiseKind::Divide, vec![1, 0]),
        ];
        let steps = compute_steps(&nodes).expect("constant / value should lower");

        assert!(steps.body.contains("rdiv_tile_init();"));
        assert!(steps.body.contains("rdiv_tile(0, 1065353216);"));
    }

    #[test]
    fn compute_steps_handles_constant_right_power_as_unary_power() {
        let mut constant = node(FusedElementwiseKind::Constant, Vec::new());
        constant.packed_value = 0x4000_4000;

        let nodes = vec![
            node(FusedElementwiseKind::Input, Vec::new()),
            constant,
            node(FusedElementwiseKind::Power, vec![0, 1]),
        ];
        let steps = compute_steps(&nodes).expect("value ** constant should lower");

        assert!(steps.body.contains("power_tile_init();"));
        assert!(steps.body.contains("power_tile(0, 1073741824);"));
    }

    #[test]
    fn compute_steps_handles_constant_right_compare_as_unary_compare() {
        let mut constant = node(FusedElementwiseKind::Constant, Vec::new());
        constant.packed_value = 0;

        let nodes = vec![
            node(FusedElementwiseKind::Input, Vec::new()),
            constant,
            node(
                FusedElementwiseKind::Compare(CompareDirection::Gt),
                vec![0, 1],
            ),
        ];
        let steps = compute_steps(&nodes).expect("value > constant should lower");

        assert!(steps.body.contains("unary_gt_tile_init();"));
        assert!(steps.body.contains("unary_gt_tile(0, 0);"));
    }

    #[test]
    fn compute_steps_reverses_constant_left_compare() {
        let mut constant = node(FusedElementwiseKind::Constant, Vec::new());
        constant.packed_value = 0;

        let nodes = vec![
            constant,
            node(FusedElementwiseKind::Input, Vec::new()),
            node(
                FusedElementwiseKind::Compare(CompareDirection::Gt),
                vec![0, 1],
            ),
        ];
        let steps = compute_steps(&nodes).expect("constant > value should lower");

        assert!(steps.body.contains("unary_lt_tile_init();"));
        assert!(steps.body.contains("unary_lt_tile(0, 0);"));
    }

    #[test]
    fn compute_steps_handles_constant_right_max_as_unary_max() {
        let mut constant = node(FusedElementwiseKind::Constant, Vec::new());
        constant.packed_value = 0;

        let nodes = vec![
            node(FusedElementwiseKind::Input, Vec::new()),
            constant,
            node(FusedElementwiseKind::Max, vec![0, 1]),
        ];
        let steps = compute_steps(&nodes).expect("max(value, constant) should lower");

        assert!(steps.body.contains("unary_max_tile_init();"));
        assert!(steps.body.contains("unary_max_tile(0, 0);"));
    }

    #[test]
    fn compute_source_only_emits_used_add_helpers() {
        let nodes = vec![
            node(FusedElementwiseKind::Input, Vec::new()),
            node(FusedElementwiseKind::Input, Vec::new()),
            node(FusedElementwiseKind::Add, vec![0, 1]),
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
            node(FusedElementwiseKind::Input, Vec::new()),
            node(FusedElementwiseKind::Input, Vec::new()),
            node(
                FusedElementwiseKind::Compare(CompareDirection::Gt),
                vec![0, 1],
            ),
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
