#[cfg(libtt_mlir_frontend)]
use crate::PJRT_Buffer_Type;
#[cfg(libtt_mlir_frontend)]
use prost::Message;
#[cfg(libtt_mlir_frontend)]
use tt_executable_proto::tt::AnalysisResult;
#[cfg(libtt_mlir_frontend)]
use tt_executable_proto::tt::Executable as ProtoExecutable;
#[cfg(libtt_mlir_frontend)]
use tt_executable_proto::tt::TensorDesc as ProtoTensorDesc;
#[cfg(libtt_mlir_frontend)]
use tt_executable_proto::tt::analysis_result::Status;
#[cfg(libtt_mlir_frontend)]
use tt_executable_proto::tt::op::Kind;
#[cfg(libtt_mlir_frontend)]
use tt_executable_proto::tt::tensor_desc::ElementType;

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
pub(crate) enum Op {
    Parameter {
        parameter_index: usize,
        output_id: u32,
    },
    Add {
        input_ids: [u32; 2],
        output_id: u32,
    },
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
}
