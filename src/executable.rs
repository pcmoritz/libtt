use crate::PJRT_Buffer_Type;
#[cfg(libtt_mlir_frontend)]
use executable_proto::tt::analysis_result::Status;
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

#[cfg(libtt_mlir_frontend)]
#[derive(Clone)]
pub(crate) struct Executable {
    pub(crate) values: Vec<ValueDesc>,
    pub(crate) ops: Vec<Op>,
    pub(crate) output_ids: Vec<u32>,
}

#[cfg(libtt_mlir_frontend)]
#[derive(Clone)]
pub(crate) struct ValueDesc {
    pub(crate) dims: Vec<i64>,
    pub(crate) element_type: PJRT_Buffer_Type,
}

#[cfg(libtt_mlir_frontend)]
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
    Matmul {
        input_ids: [u32; 2],
        output_id: u32,
        dimension_numbers: DotGeneralDimensionNumbers,
        top_k_epilogue: Option<MatmulTopKEpilogue>,
    },
    Constant {
        packed_value: u32,
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
    Compare(CompareDirection),
    Cosine,
    Sine,
    Negate,
    Exponential,
    Rsqrt,
    Convert,
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
    Mul,
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
        ProtoReduceReducer::Mul => Ok(ReduceReducer::Mul),
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

#[cfg(not(libtt_mlir_frontend))]
#[derive(Clone)]
pub(crate) struct Executable {
    pub(crate) values: Vec<ValueDesc>,
    pub(crate) ops: Vec<Op>,
    pub(crate) output_ids: Vec<u32>,
}

#[cfg(not(libtt_mlir_frontend))]
#[derive(Clone)]
pub(crate) struct ValueDesc {
    pub(crate) dims: Vec<i64>,
    pub(crate) element_type: crate::PJRT_Buffer_Type,
}

#[cfg(not(libtt_mlir_frontend))]
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
    Matmul {
        input_ids: [u32; 2],
        output_id: u32,
        dimension_numbers: DotGeneralDimensionNumbers,
        top_k_epilogue: Option<MatmulTopKEpilogue>,
    },
    Constant {
        packed_value: u32,
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
}
