#[cfg(libtt_mlir_frontend)]
use crate::PJRT_Buffer_Type;
#[cfg(libtt_mlir_frontend)]
use executable_proto::tt::analysis_result::Status;
#[cfg(libtt_mlir_frontend)]
use executable_proto::tt::compare_op::Direction as ProtoCompareDirection;
#[cfg(libtt_mlir_frontend)]
use executable_proto::tt::op::Kind;
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
    Add {
        input_ids: [u32; 2],
        output_id: u32,
    },
    Multiply {
        input_ids: [u32; 2],
        output_id: u32,
    },
    Divide {
        input_ids: [u32; 2],
        output_id: u32,
    },
    Power {
        input_ids: [u32; 2],
        output_id: u32,
    },
    Concatenate {
        input_ids: Vec<u32>,
        output_id: u32,
        dimension: u64,
    },
    Cosine {
        input_id: u32,
        output_id: u32,
    },
    Sine {
        input_id: u32,
        output_id: u32,
    },
    Convert {
        input_id: u32,
        output_id: u32,
    },
    Matmul {
        input_ids: [u32; 2],
        output_id: u32,
    },
    Max {
        input_ids: [u32; 2],
        output_id: u32,
    },
    Constant {
        packed_value: u32,
        output_id: u32,
    },
    Compare {
        input_ids: [u32; 2],
        output_id: u32,
        direction: CompareDirection,
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum CompareDirection {
    Eq,
    Ne,
    Ge,
    Gt,
    Le,
    Lt,
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
    match ProtoCompareDirection::try_from(direction)
        .map_err(|_| "TT executable compare op contains an invalid direction".to_owned())?
    {
        ProtoCompareDirection::Eq => Ok(CompareDirection::Eq),
        ProtoCompareDirection::Ne => Ok(CompareDirection::Ne),
        ProtoCompareDirection::Ge => Ok(CompareDirection::Ge),
        ProtoCompareDirection::Gt => Ok(CompareDirection::Gt),
        ProtoCompareDirection::Le => Ok(CompareDirection::Le),
        ProtoCompareDirection::Lt => Ok(CompareDirection::Lt),
    }
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
                Kind::Add(add) => Ok(Op::Add {
                    input_ids: [add.lhs_id, add.rhs_id],
                    output_id: op_desc.output_id,
                }),
                Kind::Multiply(multiply) => Ok(Op::Multiply {
                    input_ids: [multiply.lhs_id, multiply.rhs_id],
                    output_id: op_desc.output_id,
                }),
                Kind::Divide(divide) => Ok(Op::Divide {
                    input_ids: [divide.lhs_id, divide.rhs_id],
                    output_id: op_desc.output_id,
                }),
                Kind::Power(power) => Ok(Op::Power {
                    input_ids: [power.lhs_id, power.rhs_id],
                    output_id: op_desc.output_id,
                }),
                Kind::Concatenate(concatenate) => Ok(Op::Concatenate {
                    input_ids: concatenate.input_ids,
                    output_id: op_desc.output_id,
                    dimension: concatenate.dimension,
                }),
                Kind::Cosine(cosine) => Ok(Op::Cosine {
                    input_id: cosine.operand_id,
                    output_id: op_desc.output_id,
                }),
                Kind::Sine(sine) => Ok(Op::Sine {
                    input_id: sine.operand_id,
                    output_id: op_desc.output_id,
                }),
                Kind::Convert(convert) => Ok(Op::Convert {
                    input_id: convert.operand_id,
                    output_id: op_desc.output_id,
                }),
                Kind::Matmul(matmul) => Ok(Op::Matmul {
                    input_ids: [matmul.lhs_id, matmul.rhs_id],
                    output_id: op_desc.output_id,
                }),
                Kind::Max(max) => Ok(Op::Max {
                    input_ids: [max.lhs_id, max.rhs_id],
                    output_id: op_desc.output_id,
                }),
                Kind::Constant(constant) => Ok(Op::Constant {
                    packed_value: constant.packed_value,
                    output_id: op_desc.output_id,
                }),
                Kind::Compare(compare) => Ok(Op::Compare {
                    input_ids: [compare.lhs_id, compare.rhs_id],
                    output_id: op_desc.output_id,
                    direction: parse_compare_direction(compare.direction)?,
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
            }
        })
        .collect::<Result<Vec<_>, String>>()?;

    if executable.output_ids.len() != 1 {
        return Err(format!(
            "TT executable must contain exactly one output, got {}",
            executable.output_ids.len()
        ));
    }

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
    Add {
        input_ids: [u32; 2],
        output_id: u32,
    },
    Multiply {
        input_ids: [u32; 2],
        output_id: u32,
    },
    Divide {
        input_ids: [u32; 2],
        output_id: u32,
    },
    Power {
        input_ids: [u32; 2],
        output_id: u32,
    },
    Concatenate {
        input_ids: Vec<u32>,
        output_id: u32,
        dimension: u64,
    },
    Cosine {
        input_id: u32,
        output_id: u32,
    },
    Sine {
        input_id: u32,
        output_id: u32,
    },
    Convert {
        input_id: u32,
        output_id: u32,
    },
    Matmul {
        input_ids: [u32; 2],
        output_id: u32,
    },
    Max {
        input_ids: [u32; 2],
        output_id: u32,
    },
    Constant {
        packed_value: u32,
        output_id: u32,
    },
    Compare {
        input_ids: [u32; 2],
        output_id: u32,
        direction: CompareDirection,
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
}
