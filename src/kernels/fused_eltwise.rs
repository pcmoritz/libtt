use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer};
use crate::executable::{
    BitwiseBinaryKind, CompareDirection, FusedElementwiseKind, FusedElementwiseNode,
};
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
const HELPER_RAW_BITWISE: &str = r#"
template <uint32_t OP>
uint8_t raw_bitwise_u8_apply(uint8_t lhs, uint8_t rhs) {
  if constexpr (OP == 0) {
    return lhs & rhs;
  } else if constexpr (OP == 1) {
    return lhs | rhs;
  } else if constexpr (OP == 2) {
    return lhs ^ rhs;
  } else {
    uint32_t amount = static_cast<uint32_t>(rhs);
    if (amount >= 8) {
      if constexpr (OP == 5) {
        return static_cast<int8_t>(lhs) < 0 ? 0xff : 0;
      }
      return 0;
    }
    if constexpr (OP == 3) {
      return static_cast<uint8_t>(lhs << amount);
    } else if constexpr (OP == 4) {
      return static_cast<uint8_t>(lhs >> amount);
    } else {
      return static_cast<uint8_t>(static_cast<int8_t>(lhs) >> amount);
    }
  }
}

template <uint32_t OP>
void raw_bitwise_u8_tile(uint32_t lhs_cb, uint32_t rhs_cb, uint32_t output_cb) {
#ifdef TRISC_PACK
  {
    uint32_t output_addr =
        get_local_cb_interface(output_cb).fifo_wr_ptr << 4;
    mailbox_write(ckernel::ThreadId::UnpackThreadId, output_addr);
    mailbox_read(ckernel::ThreadId::UnpackThreadId);
  }
#endif
#ifdef TRISC_UNPACK
  {
    uint32_t lhs_addr =
        get_local_cb_interface(lhs_cb).fifo_rd_ptr << 4;
    uint32_t rhs_addr =
        get_local_cb_interface(rhs_cb).fifo_rd_ptr << 4;
    uint32_t output_addr = mailbox_read(ckernel::ThreadId::PackThreadId);
    volatile tt_l1_ptr uint8_t *lhs =
        reinterpret_cast<volatile tt_l1_ptr uint8_t *>(lhs_addr);
    volatile tt_l1_ptr uint8_t *rhs =
        reinterpret_cast<volatile tt_l1_ptr uint8_t *>(rhs_addr);
    volatile tt_l1_ptr uint8_t *output =
        reinterpret_cast<volatile tt_l1_ptr uint8_t *>(output_addr);
    for (uint32_t element = 0; element < 1024; ++element) {
      output[element] = raw_bitwise_u8_apply<OP>(lhs[element], rhs[element]);
    }
    mailbox_write(ckernel::ThreadId::PackThreadId, 1);
  }
#endif
}
"#;
const MAX_FUSED_INPUTS: usize = 8;
const MAX_FUSED_NODES: usize = 16;

const HEADER_ADD_INT: &str = "compute_kernel_api/add_int_sfpu.h";
const HEADER_BINARY_BITWISE: &str = "compute_kernel_api/binary_bitwise_sfpu.h";
const HEADER_BINARY_MAX_MIN: &str = "compute_kernel_api/binary_max_min.h";
const HEADER_BINARY_SHIFT: &str = "compute_kernel_api/binary_shift.h";
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
const HEADER_SUB_INT: &str = "compute_kernel_api/sub_int_sfpu.h";
const HEADER_TRIGONOMETRY: &str = "compute_kernel_api/eltwise_unary/trigonometry.h";
const HEADER_TYPECAST: &str = "compute_kernel_api/eltwise_unary/typecast.h";

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
            Self::Bitwise(_) => 2,
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
            Self::Bitwise(kind) => {
                let input_dtype = self.validate_binary_input_dtypes(node_index, input_dtypes)?;
                validate_same_output_dtype(node_index, self, input_dtype, output_dtype)?;
                if bitwise_compute(kind, input_dtype).is_none() && input_dtype != DType::UInt8 {
                    return Err(invalid_input(format!(
                        "node[{node_index}] {self:?} does not support dtype {input_dtype:?}"
                    )));
                }
                Ok(())
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

fn bitwise_compute(kind: BitwiseBinaryKind, dtype: DType) -> Option<BinaryCompute> {
    let bitwise = |tile| {
        Some(BinaryCompute {
            header: HEADER_BINARY_BITWISE,
            init: "binary_bitwise_tile_init",
            tile,
        })
    };
    let shift = |tile| {
        Some(BinaryCompute {
            header: HEADER_BINARY_SHIFT,
            init: "binary_shift_tile_init",
            tile,
        })
    };

    match (kind, dtype) {
        (BitwiseBinaryKind::And, DType::Int32) => bitwise("bitwise_and_binary_tile"),
        (BitwiseBinaryKind::And, DType::UInt32) => bitwise("bitwise_and_uint32_binary_tile"),
        (BitwiseBinaryKind::And, DType::UInt16) => bitwise("bitwise_and_uint16_binary_tile"),
        (BitwiseBinaryKind::Or, DType::Int32) => bitwise("bitwise_or_binary_tile"),
        (BitwiseBinaryKind::Or, DType::UInt32) => bitwise("bitwise_or_uint32_binary_tile"),
        (BitwiseBinaryKind::Or, DType::UInt16) => bitwise("bitwise_or_uint16_binary_tile"),
        (BitwiseBinaryKind::Xor, DType::Int32) => bitwise("bitwise_xor_binary_tile"),
        (BitwiseBinaryKind::Xor, DType::UInt32) => bitwise("bitwise_xor_uint32_binary_tile"),
        (BitwiseBinaryKind::Xor, DType::UInt16) => bitwise("bitwise_xor_uint16_binary_tile"),
        (BitwiseBinaryKind::ShiftLeft, DType::Int32) => shift("binary_left_shift_int32_tile"),
        (BitwiseBinaryKind::ShiftLeft, DType::UInt32) => shift("binary_left_shift_uint32_tile"),
        (BitwiseBinaryKind::ShiftRightLogical, DType::Int32) => {
            shift("binary_logical_right_shift_int32_tile")
        }
        (BitwiseBinaryKind::ShiftRightLogical, DType::UInt32) => {
            shift("binary_logical_right_shift_uint32_tile")
        }
        (BitwiseBinaryKind::ShiftRightArithmetic, DType::Int32) => {
            shift("binary_right_shift_int32_tile")
        }
        (BitwiseBinaryKind::ShiftRightArithmetic, DType::UInt32) => {
            shift("binary_right_shift_uint32_tile")
        }
        _ => None,
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
    tile_count: u32,
    output_dtype: DType,
    nodes: Vec<FusedElementwiseNode>,
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

fn node_dtype(node: &FusedElementwiseNode) -> io::Result<DType> {
    pjrt_buffer_type_to_dtype(node.element_type)
}

pub(crate) fn eltwise(
    device: &mut Device,
    external_inputs: &[&DramBuffer],
    nodes: &[FusedElementwiseNode],
    shape: &[usize],
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    let input_reads = validate_and_collect_inputs(external_inputs, nodes, shape)?;

    let output_tiles = tiled_shape_tile_count(shape)?;
    let tile_count = u32_arg(output_tiles, "tile count")?;
    let cores = select_worker_cores(device.cores_ref(), output_tiles)?;
    let output_dtype = node_dtype(&nodes[nodes.len() - 1])?;
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
    if input_reads.len() > MAX_FUSED_INPUTS {
        return Err(invalid_input(format!(
            "fused eltwise supports at most {MAX_FUSED_INPUTS} leaf inputs, got {}",
            input_reads.len()
        )));
    }
    Ok(input_reads)
}

fn fused_eltwise_program(key: FusedEltwiseProgramKey) -> io::Result<Program> {
    let (node_cbs, intermediate_cbs) = cb_plan(&key.nodes)?;
    let leaf_nodes = fused_leaf_nodes(&key.nodes, &node_cbs)?;
    let input_count = leaf_nodes
        .iter()
        .filter(|(_, node)| node.kind == FusedElementwiseKind::Input)
        .count();
    let reader_dynamic_indices: Vec<usize> = (0..input_count).collect();

    let mut runtime_args = RuntimeArgsBuilder::new(0, vec![0], reader_dynamic_indices, Vec::new());
    for (core_index, &core) in key.cores.iter().enumerate() {
        let (offset, n_tiles) = split_tile_range(key.tile_count, core_index, key.cores.len())?;
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
    // Multi-stage fusions reuse DST across several compute/pack sections in
    // one kernel; full sync avoids half-DST ping-pong races between stages.
    let fused_compute_ops = key
        .nodes
        .iter()
        .filter(|node| node.kind.arity() > 0)
        .count();
    if fused_compute_ops > 1 {
        for cb in &mut cbs {
            cb.tiles = cb.tiles.max(4);
        }
    }

    let mut dst_accum_mode = matches!(
        key.output_dtype,
        DType::Float32 | DType::Int32 | DType::UInt32
    );
    for node in &key.nodes {
        if matches!(
            node_dtype(node)?,
            DType::Float32 | DType::Int32 | DType::UInt32
        ) {
            dst_accum_mode = true;
            break;
        }
    }

    Ok(Program {
        reader_kernel: reader_source(&leaf_nodes, input_count)?,
        compute_kernel: compute_source(&key)?,
        writer_kernel: WRITER.to_owned(),
        compile: CompileConfig {
            cbs,
            dst_accum_mode,
            dst_full_sync: fused_compute_ops > 1,
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
) -> io::Result<String> {
    let mut arg_loads = String::new();
    let mut addr_gens = String::new();
    let mut reserves = String::new();
    let mut reads = String::new();
    let mut broadcasts = String::new();
    let mut pushes = String::new();
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
                let tile_id = if node.single_tile_broadcast {
                    "0"
                } else {
                    "offset + i"
                };
                writeln!(
                    reads,
                    "    noc_async_read_tile({tile_id}, input_{input_arg_index}, get_write_ptr(cb_leaf_{leaf_index}));"
                )
                .unwrap();
                if node.single_tile_broadcast {
                    let bytes = element_bytes(node_dtype(node)?);
                    let helper = if node.packed_value == 1 {
                        "replicate_first_column"
                    } else {
                        "replicate_first_element"
                    };
                    writeln!(broadcasts, "    {helper}(cb_leaf_{leaf_index}, {bytes});").unwrap();
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

    Ok(format!(
        "#include <cstdint>\n\
         \n\
         namespace {{\n\
         constexpr uint32_t TILE_R = 32;\n\
         constexpr uint32_t TILE_C = 32;\n\
         constexpr uint32_t FACE_R = 16;\n\
         constexpr uint32_t FACE_C = 16;\n\
         uint32_t tile_element_index(uint32_t row, uint32_t col) {{\n\
           uint32_t face_row = row / FACE_R;\n\
           uint32_t face_col = col / FACE_C;\n\
           uint32_t row_in_face = row % FACE_R;\n\
           uint32_t col_in_face = col % FACE_C;\n\
           return ((face_row * 2 + face_col) * FACE_R * FACE_C) + row_in_face * FACE_C + col_in_face;\n\
         }}\n\
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
         void replicate_first_column(uint32_t cb, uint32_t element_bytes) {{\n\
           uint32_t l1_addr = get_write_ptr(cb);\n\
           if (element_bytes == 4) {{\n\
             volatile tt_l1_ptr uint32_t *ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(l1_addr);\n\
             for (uint32_t row = 0; row < TILE_R; ++row) {{\n\
               uint32_t value = ptr[tile_element_index(row, 0)];\n\
               for (uint32_t col = 1; col < TILE_C; ++col) {{\n\
                 ptr[tile_element_index(row, col)] = value;\n\
               }}\n\
             }}\n\
           }} else if (element_bytes == 2) {{\n\
             volatile tt_l1_ptr uint16_t *ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t *>(l1_addr);\n\
             for (uint32_t row = 0; row < TILE_R; ++row) {{\n\
               uint16_t value = ptr[tile_element_index(row, 0)];\n\
               for (uint32_t col = 1; col < TILE_C; ++col) {{\n\
                 ptr[tile_element_index(row, col)] = value;\n\
               }}\n\
             }}\n\
           }} else {{\n\
             volatile tt_l1_ptr uint8_t *ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t *>(l1_addr);\n\
             for (uint32_t row = 0; row < TILE_R; ++row) {{\n\
               uint8_t value = ptr[tile_element_index(row, 0)];\n\
               for (uint32_t col = 1; col < TILE_C; ++col) {{\n\
                 ptr[tile_element_index(row, col)] = value;\n\
               }}\n\
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
         {reserves}\
         {reads}\
             noc_async_read_barrier();\n\
         {broadcasts}\
         {pushes}\
           }}\n\
         }}\n",
        input_count + 1
    ))
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
    raw_bitwise: bool,
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

    fn add_raw_bitwise(&mut self) {
        self.raw_bitwise = true;
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
        if self.raw_bitwise {
            helpers.push_str(HELPER_RAW_BITWISE);
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
    RawBitwise(RawBitwiseSpec),
    Compare(CompareSpec),
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
struct RawBitwiseSpec {
    lhs: usize,
    rhs: usize,
    kind: BitwiseBinaryKind,
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
            Lowering::RawBitwise(spec) => emit_raw_bitwise(&mut ctx, index, spec)?,
            Lowering::Compare(spec) => emit_compare(&mut ctx, index, spec)?,
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
            if let FusedElementwiseKind::Bitwise(kind) = node.kind {
                let dtype = node_dtype(node)?;
                if let Some(binary) = bitwise_compute(kind, dtype) {
                    return Ok(Lowering::Binary(BinarySpec {
                        lhs,
                        rhs,
                        op: BinaryLowering::Tile(binary),
                    }));
                }
                if dtype != DType::UInt8 {
                    return Err(invalid_input(format!(
                        "missing bitwise lowering for {:?} with dtype {dtype:?}",
                        node.kind
                    )));
                }
                return Ok(Lowering::RawBitwise(RawBitwiseSpec { lhs, rhs, kind }));
            }
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
        _ => unreachable!("fused eltwise op arity is limited to 0, 1, or 2"),
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

fn emit_raw_bitwise(
    ctx: &mut ComputeEmitContext<'_>,
    index: usize,
    spec: RawBitwiseSpec,
) -> io::Result<()> {
    let lhs_cb = ctx.cb_for_node(spec.lhs)?;
    let rhs_cb = ctx.cb_for_node(spec.rhs)?;
    let output_cb = ctx.cb_for_node(index)?;
    append_waits(&mut ctx.body, &[lhs_cb, rhs_cb]);
    ctx.features.add_raw_bitwise();
    writeln!(
        ctx.body,
        "    cb_reserve_back(tt::CBIndex::c_{output_cb}, 1);"
    )
    .unwrap();
    writeln!(
        ctx.body,
        "    raw_bitwise_u8_tile<{}>(tt::CBIndex::c_{lhs_cb}, tt::CBIndex::c_{rhs_cb}, tt::CBIndex::c_{output_cb});",
        raw_bitwise_op_value(spec.kind)
    )
    .unwrap();
    append_pop_consumed_and_push(
        &mut ctx.body,
        output_cb,
        &[spec.lhs, spec.rhs],
        ctx.node_cbs,
        ctx.remaining_uses,
    )
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
    append_pop_consumed_and_push(body, output_cb, input_nodes, node_cbs, remaining_uses)
}

fn append_pop_consumed_and_push(
    body: &mut String,
    output_cb: u32,
    input_nodes: &[usize],
    node_cbs: &[Option<u32>],
    remaining_uses: &mut [u32],
) -> io::Result<()> {
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

fn element_bytes(dtype: DType) -> usize {
    match dtype {
        DType::Float32 | DType::Int32 | DType::UInt32 => 4,
        DType::Float16 | DType::Float16B | DType::UInt16 => 2,
        DType::Int8 | DType::UInt8 => 1,
    }
}

fn raw_bitwise_op_value(kind: BitwiseBinaryKind) -> u32 {
    match kind {
        BitwiseBinaryKind::And => 0,
        BitwiseBinaryKind::Or => 1,
        BitwiseBinaryKind::Xor => 2,
        BitwiseBinaryKind::ShiftLeft => 3,
        BitwiseBinaryKind::ShiftRightLogical => 4,
        BitwiseBinaryKind::ShiftRightArithmetic => 5,
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
        }
    }

    fn program_key(nodes: Vec<FusedElementwiseNode>) -> FusedEltwiseProgramKey {
        FusedEltwiseProgramKey {
            cores: Vec::new(),
            tile_count: 1,
            output_dtype: node_dtype(nodes.last().expect("test nodes must not be empty"))
                .expect("test node dtype must be valid"),
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
    fn validate_allows_constant_only_fusion() {
        let mut lhs = node(FusedElementwiseKind::Constant, Vec::new());
        lhs.packed_value = 0x3f80_3f80;
        let mut rhs = node(FusedElementwiseKind::Constant, Vec::new());
        rhs.packed_value = 0x4000_4000;

        let nodes = vec![lhs, rhs, node(FusedElementwiseKind::Add, vec![0, 1])];
        let inputs = validate_and_collect_inputs(&[], &nodes, &[32, 32])
            .expect("constant-only fused op should not require external inputs");

        assert!(inputs.is_empty());
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
