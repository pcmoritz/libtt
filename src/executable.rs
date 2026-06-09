use crate::PJRT_Buffer_Type;
#[cfg(libtt_mlir_frontend)]
use executable_proto::tt::analysis_result::Status;
#[cfg(libtt_mlir_frontend)]
use executable_proto::tt::bitwise_binary_op::Kind as ProtoBitwiseBinaryKind;
#[cfg(libtt_mlir_frontend)]
use executable_proto::tt::fused_elementwise_op::node::CompareDirection as ProtoFusedElementwiseCompareDirection;
#[cfg(libtt_mlir_frontend)]
use executable_proto::tt::fused_elementwise_op::node::Kind as ProtoFusedElementwiseKind;
#[cfg(libtt_mlir_frontend)]
use executable_proto::tt::op::Kind;
#[cfg(libtt_mlir_frontend)]
use executable_proto::tt::reduce_op::Reducer as ProtoReduceReducer;
#[cfg(libtt_mlir_frontend)]
use executable_proto::tt::tensor_desc::ElementType;
#[cfg(libtt_mlir_frontend)]
use executable_proto::tt::AnalysisResult;
#[cfg(libtt_mlir_frontend)]
use executable_proto::tt::Executable as ProtoExecutable;
#[cfg(libtt_mlir_frontend)]
use executable_proto::tt::TensorDesc as ProtoTensorDesc;
#[cfg(libtt_mlir_frontend)]
use prost::Message;

#[derive(Clone)]
pub(crate) struct Executable {
    pub(crate) values: Vec<ValueDesc>,
    pub(crate) ops: Vec<Op>,
    pub(crate) output_ids: Vec<u32>,
}

#[derive(Clone)]
pub(crate) struct ValueDesc {
    pub(crate) dims: Vec<i64>,
    pub(crate) element_type: PJRT_Buffer_Type,
}

#[derive(Clone)]
#[allow(dead_code)]
pub(crate) enum Op {
    Parameter {
        parameter_index: usize,
        output_id: u32,
    },
    Concatenate {
        input_ids: Vec<u32>,
        output_id: u32,
        dimension: u64,
    },
    Reshape {
        input_id: u32,
        output_id: u32,
    },
    Slice {
        input_id: u32,
        output_id: u32,
        start_indices: Vec<i64>,
        limit_indices: Vec<i64>,
        strides: Vec<i64>,
    },
    Transpose {
        input_id: u32,
        output_id: u32,
        permutation: Vec<i64>,
    },
    CustomCall {
        input_ids: Vec<u32>,
        output_id: u32,
        call_target_name: String,
        has_side_effect: bool,
    },
    Reduce {
        input_ids: Vec<u32>,
        init_value_ids: Vec<u32>,
        output_id: u32,
        dimensions: Vec<i64>,
        reducer: ReduceReducer,
    },
    ReduceWindow {
        input_ids: Vec<u32>,
        init_value_ids: Vec<u32>,
        output_id: u32,
        attributes: ReduceWindowAttributes,
        reducer: ReduceReducer,
    },
    Matmul {
        input_ids: [u32; 2],
        output_id: u32,
        dimension_numbers: DotGeneralDimensionNumbers,
        top_k_epilogue: Option<MatmulTopKEpilogue>,
    },
    Constant {
        packed_value: u32,
        data: Vec<u8>,
        output_id: u32,
    },
    Select {
        input_ids: [u32; 3],
        output_id: u32,
    },
    BroadcastInDim {
        input_id: u32,
        output_id: u32,
        broadcast_dimensions: Vec<i64>,
    },
    Gather {
        input_ids: [u32; 2],
        output_id: u32,
        dimension_numbers: GatherDimensionNumbers,
        slice_sizes: Vec<i64>,
        indices_are_sorted: bool,
    },
    Scatter {
        input_ids: [u32; 3],
        output_id: u32,
        dimension_numbers: ScatterDimensionNumbers,
        indices_are_sorted: bool,
        unique_indices: bool,
    },
    Iota {
        output_id: u32,
        iota_dimension: u64,
    },
    TopK {
        input_id: u32,
        values_id: u32,
        indices_id: u32,
        k: u32,
    },
    FusedElementwise {
        input_ids: Vec<u32>,
        output_id: u32,
        nodes: Vec<FusedElementwiseNode>,
    },
    SdpaDecode {
        input_ids: [u32; 5],
        output_id: u32,
        scale_bf16_packed: u32,
    },
    RmsNorm {
        input_ids: [u32; 2],
        output_id: u32,
        scale_bits: u32,
        bias_bits: u32,
    },
    Rope {
        input_ids: [u32; 3],
        output_id: u32,
    },
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) struct FusedElementwiseNode {
    pub(crate) kind: FusedElementwiseKind,
    pub(crate) input_nodes: Vec<u32>,
    pub(crate) input_index: u32,
    pub(crate) packed_value: u32,
    pub(crate) element_type: PJRT_Buffer_Type,
    pub(crate) single_tile_broadcast: bool,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum FusedElementwiseKind {
    Input,
    Constant,
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Max,
    Bitwise(BitwiseBinaryKind),
    Compare(CompareDirection),
    Cosine,
    Sine,
    Negate,
    Exponential,
    Rsqrt,
    Log,
    Convert,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum BitwiseBinaryKind {
    And,
    Or,
    Xor,
    ShiftLeft,
    ShiftRightLogical,
    ShiftRightArithmetic,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum CompareDirection {
    Eq,
    Ne,
    Ge,
    Gt,
    Le,
    Lt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum ReduceReducer {
    Add,
    Max,
    Min,
    Mul,
    And,
    Or,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) struct ReduceWindowAttributes {
    pub(crate) window_dimensions: Vec<i64>,
    pub(crate) window_strides: Vec<i64>,
    pub(crate) base_dilations: Vec<i64>,
    pub(crate) window_dilations: Vec<i64>,
    pub(crate) padding_low: Vec<i64>,
    pub(crate) padding_high: Vec<i64>,
}

#[derive(Clone)]
#[allow(dead_code)]
pub(crate) struct GatherDimensionNumbers {
    pub(crate) offset_dims: Vec<i64>,
    pub(crate) collapsed_slice_dims: Vec<i64>,
    pub(crate) operand_batching_dims: Vec<i64>,
    pub(crate) start_indices_batching_dims: Vec<i64>,
    pub(crate) start_index_map: Vec<i64>,
    pub(crate) index_vector_dim: i64,
}

#[derive(Clone)]
#[allow(dead_code)]
pub(crate) struct ScatterDimensionNumbers {
    pub(crate) update_window_dims: Vec<i64>,
    pub(crate) inserted_window_dims: Vec<i64>,
    pub(crate) input_batching_dims: Vec<i64>,
    pub(crate) scatter_indices_batching_dims: Vec<i64>,
    pub(crate) scatter_dims_to_operand_dims: Vec<i64>,
    pub(crate) index_vector_dim: i64,
}

#[derive(Clone)]
#[allow(dead_code)]
pub(crate) struct DotGeneralDimensionNumbers {
    pub(crate) lhs_batching_dimensions: Vec<i64>,
    pub(crate) rhs_batching_dimensions: Vec<i64>,
    pub(crate) lhs_contracting_dimensions: Vec<i64>,
    pub(crate) rhs_contracting_dimensions: Vec<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) struct MatmulTopKEpilogue {
    pub(crate) matmul_output_id: u32,
    pub(crate) indices_id: u32,
    pub(crate) k: u32,
}

#[cfg(libtt_mlir_frontend)]
pub(crate) struct Analysis {
    pub(crate) status: Status,
    pub(crate) error_message: String,
    pub(crate) outputs: Vec<ValueDesc>,
    pub(crate) executable: Option<Executable>,
}

#[cfg(libtt_mlir_frontend)]
fn map_element_type(element_type: i32) -> Result<PJRT_Buffer_Type, String> {
    match ElementType::try_from(element_type)
        .map_err(|_| "TT executable contains an invalid tensor element type".to_owned())?
    {
        ElementType::Unknown => Err("TT executable contains an unknown tensor element type".into()),
        ElementType::Bf16 => Ok(PJRT_Buffer_Type::PJRT_Buffer_Type_BF16),
        ElementType::F16 => Ok(PJRT_Buffer_Type::PJRT_Buffer_Type_F16),
        ElementType::F32 => Ok(PJRT_Buffer_Type::PJRT_Buffer_Type_F32),
        ElementType::U32 => Ok(PJRT_Buffer_Type::PJRT_Buffer_Type_U32),
        ElementType::U16 => Ok(PJRT_Buffer_Type::PJRT_Buffer_Type_U16),
        ElementType::U8 => Ok(PJRT_Buffer_Type::PJRT_Buffer_Type_U8),
        ElementType::S32 => Ok(PJRT_Buffer_Type::PJRT_Buffer_Type_S32),
        ElementType::S8 => Ok(PJRT_Buffer_Type::PJRT_Buffer_Type_S8),
        ElementType::Pred => Ok(PJRT_Buffer_Type::PJRT_Buffer_Type_PRED),
    }
}

#[cfg(libtt_mlir_frontend)]
fn parse_tensor_desc(tensor: ProtoTensorDesc) -> Result<ValueDesc, String> {
    Ok(ValueDesc {
        dims: tensor.dims,
        element_type: map_element_type(tensor.element_type)?,
    })
}

#[cfg(libtt_mlir_frontend)]
fn parse_compare_direction(direction: i32) -> Result<CompareDirection, String> {
    match ProtoFusedElementwiseCompareDirection::try_from(direction).map_err(|_| {
        "TT executable fused elementwise compare node contains an invalid direction".to_owned()
    })? {
        ProtoFusedElementwiseCompareDirection::DirectionEq => Ok(CompareDirection::Eq),
        ProtoFusedElementwiseCompareDirection::DirectionNe => Ok(CompareDirection::Ne),
        ProtoFusedElementwiseCompareDirection::DirectionGe => Ok(CompareDirection::Ge),
        ProtoFusedElementwiseCompareDirection::DirectionGt => Ok(CompareDirection::Gt),
        ProtoFusedElementwiseCompareDirection::DirectionLe => Ok(CompareDirection::Le),
        ProtoFusedElementwiseCompareDirection::DirectionLt => Ok(CompareDirection::Lt),
    }
}

#[cfg(libtt_mlir_frontend)]
fn parse_reduce_reducer(reducer: i32) -> Result<ReduceReducer, String> {
    match ProtoReduceReducer::try_from(reducer)
        .map_err(|_| "TT executable reduce op contains an invalid reducer".to_owned())?
    {
        ProtoReduceReducer::Add => Ok(ReduceReducer::Add),
        ProtoReduceReducer::Max => Ok(ReduceReducer::Max),
        ProtoReduceReducer::Min => Ok(ReduceReducer::Min),
        ProtoReduceReducer::Mul => Ok(ReduceReducer::Mul),
        ProtoReduceReducer::And => Ok(ReduceReducer::And),
        ProtoReduceReducer::Or => Ok(ReduceReducer::Or),
    }
}

#[cfg(libtt_mlir_frontend)]
fn parse_bitwise_binary_kind(kind: ProtoBitwiseBinaryKind) -> Result<BitwiseBinaryKind, String> {
    match kind {
        ProtoBitwiseBinaryKind::And => Ok(BitwiseBinaryKind::And),
        ProtoBitwiseBinaryKind::Or => Ok(BitwiseBinaryKind::Or),
        ProtoBitwiseBinaryKind::Xor => Ok(BitwiseBinaryKind::Xor),
        ProtoBitwiseBinaryKind::ShiftLeft => Ok(BitwiseBinaryKind::ShiftLeft),
        ProtoBitwiseBinaryKind::ShiftRightLogical => Ok(BitwiseBinaryKind::ShiftRightLogical),
        ProtoBitwiseBinaryKind::ShiftRightArithmetic => Ok(BitwiseBinaryKind::ShiftRightArithmetic),
    }
}

#[cfg(libtt_mlir_frontend)]
fn value_element_type(
    values: &[ValueDesc],
    value_id: u32,
    label: &str,
) -> Result<PJRT_Buffer_Type, String> {
    values
        .get(value_id as usize)
        .map(|value| value.element_type)
        .ok_or_else(|| {
            format!("TT executable bitwise {label} value id {value_id} is out of bounds")
        })
}

#[cfg(libtt_mlir_frontend)]
fn fused_input_node(input_index: u32, element_type: PJRT_Buffer_Type) -> FusedElementwiseNode {
    FusedElementwiseNode {
        kind: FusedElementwiseKind::Input,
        input_nodes: Vec::new(),
        input_index,
        packed_value: 0,
        element_type,
        single_tile_broadcast: false,
    }
}

#[cfg(libtt_mlir_frontend)]
fn fused_bitwise_node(
    kind: BitwiseBinaryKind,
    element_type: PJRT_Buffer_Type,
) -> FusedElementwiseNode {
    FusedElementwiseNode {
        kind: FusedElementwiseKind::Bitwise(kind),
        input_nodes: vec![0, 1],
        input_index: 0,
        packed_value: 0,
        element_type,
        single_tile_broadcast: false,
    }
}

#[cfg(libtt_mlir_frontend)]
fn parse_fused_elementwise_kind(
    kind: i32,
    compare_direction: i32,
) -> Result<FusedElementwiseKind, String> {
    match ProtoFusedElementwiseKind::try_from(kind)
        .map_err(|_| "TT executable fused elementwise op contains an invalid kind".to_owned())?
    {
        ProtoFusedElementwiseKind::Input => Ok(FusedElementwiseKind::Input),
        ProtoFusedElementwiseKind::Constant => Ok(FusedElementwiseKind::Constant),
        ProtoFusedElementwiseKind::Add => Ok(FusedElementwiseKind::Add),
        ProtoFusedElementwiseKind::Subtract => Ok(FusedElementwiseKind::Subtract),
        ProtoFusedElementwiseKind::Multiply => Ok(FusedElementwiseKind::Multiply),
        ProtoFusedElementwiseKind::Divide => Ok(FusedElementwiseKind::Divide),
        ProtoFusedElementwiseKind::Power => Ok(FusedElementwiseKind::Power),
        ProtoFusedElementwiseKind::Max => Ok(FusedElementwiseKind::Max),
        ProtoFusedElementwiseKind::Compare => Ok(FusedElementwiseKind::Compare(
            parse_compare_direction(compare_direction)?,
        )),
        ProtoFusedElementwiseKind::Cosine => Ok(FusedElementwiseKind::Cosine),
        ProtoFusedElementwiseKind::Sine => Ok(FusedElementwiseKind::Sine),
        ProtoFusedElementwiseKind::Negate => Ok(FusedElementwiseKind::Negate),
        ProtoFusedElementwiseKind::Exponential => Ok(FusedElementwiseKind::Exponential),
        ProtoFusedElementwiseKind::Rsqrt => Ok(FusedElementwiseKind::Rsqrt),
        ProtoFusedElementwiseKind::Log => Ok(FusedElementwiseKind::Log),
        ProtoFusedElementwiseKind::Convert => Ok(FusedElementwiseKind::Convert),
    }
}

#[cfg(libtt_mlir_frontend)]
fn parse_fused_elementwise_node(
    node: executable_proto::tt::fused_elementwise_op::Node,
) -> Result<FusedElementwiseNode, String> {
    Ok(FusedElementwiseNode {
        kind: parse_fused_elementwise_kind(node.kind, node.compare_direction)?,
        input_nodes: node.input_nodes,
        input_index: node.input_index,
        packed_value: node.packed_value,
        element_type: map_element_type(node.element_type)?,
        single_tile_broadcast: node.single_tile_broadcast,
    })
}

#[cfg(libtt_mlir_frontend)]
pub(crate) fn parse_proto(executable: ProtoExecutable) -> Result<Executable, String> {
    let values = executable
        .values
        .into_iter()
        .map(|value| {
            let tensor = value
                .tensor
                .ok_or_else(|| "TT executable value is missing tensor metadata".to_owned())?;
            parse_tensor_desc(tensor)
        })
        .collect::<Result<Vec<_>, String>>()?;

    let ops = executable
        .ops
        .into_iter()
        .map(|op_desc| {
            match op_desc
                .kind
                .ok_or_else(|| "TT executable op is missing kind".to_owned())?
            {
                Kind::Parameter(parameter) => Ok(Op::Parameter {
                    parameter_index: parameter.parameter_index as usize,
                    output_id: op_desc.output_id,
                }),
                Kind::Concatenate(concatenate) => Ok(Op::Concatenate {
                    input_ids: concatenate.input_ids,
                    output_id: op_desc.output_id,
                    dimension: concatenate.dimension,
                }),
                Kind::Reshape(reshape) => Ok(Op::Reshape {
                    input_id: reshape.operand_id,
                    output_id: op_desc.output_id,
                }),
                Kind::Slice(slice) => Ok(Op::Slice {
                    input_id: slice.operand_id,
                    output_id: op_desc.output_id,
                    start_indices: slice.start_indices,
                    limit_indices: slice.limit_indices,
                    strides: slice.strides,
                }),
                Kind::Transpose(transpose) => Ok(Op::Transpose {
                    input_id: transpose.operand_id,
                    output_id: op_desc.output_id,
                    permutation: transpose.permutation,
                }),
                Kind::CustomCall(custom_call) => Ok(Op::CustomCall {
                    input_ids: custom_call.input_ids,
                    output_id: op_desc.output_id,
                    call_target_name: custom_call.call_target_name,
                    has_side_effect: custom_call.has_side_effect,
                }),
                Kind::Reduce(reduce) => Ok(Op::Reduce {
                    input_ids: reduce.input_ids,
                    init_value_ids: reduce.init_value_ids,
                    output_id: op_desc.output_id,
                    dimensions: reduce.dimensions,
                    reducer: parse_reduce_reducer(reduce.reducer)?,
                }),
                Kind::ReduceWindow(reduce_window) => Ok(Op::ReduceWindow {
                    input_ids: reduce_window.input_ids,
                    init_value_ids: reduce_window.init_value_ids,
                    output_id: op_desc.output_id,
                    attributes: ReduceWindowAttributes {
                        window_dimensions: reduce_window.window_dimensions,
                        window_strides: reduce_window.window_strides,
                        base_dilations: reduce_window.base_dilations,
                        window_dilations: reduce_window.window_dilations,
                        padding_low: reduce_window.padding_low,
                        padding_high: reduce_window.padding_high,
                    },
                    reducer: parse_reduce_reducer(reduce_window.reducer)?,
                }),
                Kind::Matmul(matmul) => Ok(Op::Matmul {
                    input_ids: [matmul.lhs_id, matmul.rhs_id],
                    output_id: op_desc.output_id,
                    dimension_numbers: DotGeneralDimensionNumbers {
                        lhs_batching_dimensions: matmul.lhs_batching_dimensions,
                        rhs_batching_dimensions: matmul.rhs_batching_dimensions,
                        lhs_contracting_dimensions: matmul.lhs_contracting_dimensions,
                        rhs_contracting_dimensions: matmul.rhs_contracting_dimensions,
                    },
                    top_k_epilogue: matmul.top_k_epilogue.map(|epilogue| MatmulTopKEpilogue {
                        matmul_output_id: epilogue.matmul_output_id,
                        indices_id: epilogue.indices_id,
                        k: epilogue.k,
                    }),
                }),
                Kind::Constant(constant) => Ok(Op::Constant {
                    packed_value: constant.packed_value,
                    data: constant.data,
                    output_id: op_desc.output_id,
                }),
                Kind::Select(select) => Ok(Op::Select {
                    input_ids: [select.pred_id, select.on_true_id, select.on_false_id],
                    output_id: op_desc.output_id,
                }),
                Kind::BroadcastInDim(broadcast) => Ok(Op::BroadcastInDim {
                    input_id: broadcast.operand_id,
                    output_id: op_desc.output_id,
                    broadcast_dimensions: broadcast.broadcast_dimensions,
                }),
                Kind::Gather(gather) => Ok(Op::Gather {
                    input_ids: [gather.operand_id, gather.start_indices_id],
                    output_id: op_desc.output_id,
                    dimension_numbers: GatherDimensionNumbers {
                        offset_dims: gather.offset_dims,
                        collapsed_slice_dims: gather.collapsed_slice_dims,
                        operand_batching_dims: gather.operand_batching_dims,
                        start_indices_batching_dims: gather.start_indices_batching_dims,
                        start_index_map: gather.start_index_map,
                        index_vector_dim: gather.index_vector_dim,
                    },
                    slice_sizes: gather.slice_sizes,
                    indices_are_sorted: gather.indices_are_sorted,
                }),
                Kind::Scatter(scatter) => Ok(Op::Scatter {
                    input_ids: [
                        scatter.operand_id,
                        scatter.start_indices_id,
                        scatter.updates_id,
                    ],
                    output_id: op_desc.output_id,
                    dimension_numbers: ScatterDimensionNumbers {
                        update_window_dims: scatter.update_window_dims,
                        inserted_window_dims: scatter.inserted_window_dims,
                        input_batching_dims: scatter.input_batching_dims,
                        scatter_indices_batching_dims: scatter.scatter_indices_batching_dims,
                        scatter_dims_to_operand_dims: scatter.scatter_dims_to_operand_dims,
                        index_vector_dim: scatter.index_vector_dim,
                    },
                    indices_are_sorted: scatter.indices_are_sorted,
                    unique_indices: scatter.unique_indices,
                }),
                Kind::BitwiseBinary(bitwise) => {
                    let lhs_id = bitwise.lhs_id;
                    let rhs_id = bitwise.rhs_id;
                    let output_id = op_desc.output_id;
                    let lhs_element_type = value_element_type(&values, lhs_id, "lhs")?;
                    let rhs_element_type = value_element_type(&values, rhs_id, "rhs")?;
                    let output_element_type = value_element_type(&values, output_id, "output")?;
                    Ok(Op::FusedElementwise {
                        input_ids: vec![lhs_id, rhs_id],
                        output_id,
                        nodes: vec![
                            fused_input_node(0, lhs_element_type),
                            fused_input_node(1, rhs_element_type),
                            fused_bitwise_node(
                                parse_bitwise_binary_kind(bitwise.kind())?,
                                output_element_type,
                            ),
                        ],
                    })
                }
                Kind::Iota(iota) => Ok(Op::Iota {
                    output_id: op_desc.output_id,
                    iota_dimension: iota.iota_dimension,
                }),
                Kind::TopK(top_k) => Ok(Op::TopK {
                    input_id: top_k.operand_id,
                    values_id: op_desc.output_id,
                    indices_id: top_k.indices_id,
                    k: top_k.k,
                }),
                Kind::FusedElementwise(fused) => Ok(Op::FusedElementwise {
                    input_ids: fused.input_ids,
                    output_id: op_desc.output_id,
                    nodes: fused
                        .nodes
                        .into_iter()
                        .map(parse_fused_elementwise_node)
                        .collect::<Result<Vec<_>, String>>()?,
                }),
                Kind::SdpaDecode(sdpa_decode) => Ok(Op::SdpaDecode {
                    input_ids: [
                        sdpa_decode.q_id,
                        sdpa_decode.k_id,
                        sdpa_decode.v_id,
                        sdpa_decode.seq_lens_id,
                        sdpa_decode.loc_id,
                    ],
                    output_id: op_desc.output_id,
                    scale_bf16_packed: sdpa_decode.scale_bf16_packed,
                }),
                Kind::RmsNorm(rms_norm) => Ok(Op::RmsNorm {
                    input_ids: [rms_norm.input_id, rms_norm.weight_id],
                    output_id: op_desc.output_id,
                    scale_bits: rms_norm.scale_bits,
                    bias_bits: rms_norm.bias_bits,
                }),
                Kind::Rope(rope) => Ok(Op::Rope {
                    input_ids: [rope.input_id, rope.cos_id, rope.sin_id],
                    output_id: op_desc.output_id,
                }),
            }
        })
        .collect::<Result<Vec<_>, String>>()?;

    Ok(Executable {
        values,
        ops,
        output_ids: executable.output_ids,
    })
}

#[cfg(libtt_mlir_frontend)]
pub(crate) fn parse_analysis(bytes: &[u8]) -> Result<Analysis, String> {
    let analysis = AnalysisResult::decode(bytes)
        .map_err(|err| format!("failed to parse TT MLIR analysis result: {err}"))?;
    let status = Status::try_from(analysis.status)
        .map_err(|_| "TT MLIR analysis result contains an invalid status".to_owned())?;
    let outputs = analysis
        .outputs
        .into_iter()
        .map(parse_tensor_desc)
        .collect::<Result<Vec<_>, String>>()?;

    let executable = if let Some(executable_proto) = analysis.executable {
        Some(parse_proto(executable_proto)?)
    } else {
        None
    };

    Ok(Analysis {
        status,
        error_message: analysis.error_message,
        outputs,
        executable,
    })
}
