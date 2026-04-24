#[cfg(libtt_mlir_frontend)]
use crate::PJRT_Buffer_Type;
#[cfg(libtt_mlir_frontend)]
use prost::Message;
#[cfg(libtt_mlir_frontend)]
use tt_executable_proto::tt::TtExecutableV1;
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
pub(crate) fn parse(bytes: &[u8]) -> Result<Executable, String> {
    let executable = TtExecutableV1::decode(bytes)
        .map_err(|err| format!("failed to parse TT executable: {err}"))?;

    let values = executable
        .values
        .into_iter()
        .map(|value| {
            let tensor = value
                .tensor
                .ok_or_else(|| "TT executable value is missing tensor metadata".to_owned())?;
            let element_type = match ElementType::try_from(tensor.element_type)
                .map_err(|_| "TT executable contains an invalid tensor element type".to_owned())?
            {
                ElementType::Unknown => {
                    return Err("TT executable contains an unknown tensor element type".into());
                }
                ElementType::Bf16 => PJRT_Buffer_Type::PJRT_Buffer_Type_BF16,
                ElementType::F16 => PJRT_Buffer_Type::PJRT_Buffer_Type_F16,
                ElementType::F32 => PJRT_Buffer_Type::PJRT_Buffer_Type_F32,
                ElementType::U32 => PJRT_Buffer_Type::PJRT_Buffer_Type_U32,
                ElementType::U16 => PJRT_Buffer_Type::PJRT_Buffer_Type_U16,
                ElementType::U8 => PJRT_Buffer_Type::PJRT_Buffer_Type_U8,
                ElementType::S32 => PJRT_Buffer_Type::PJRT_Buffer_Type_S32,
                ElementType::S8 => PJRT_Buffer_Type::PJRT_Buffer_Type_S8,
            };
            Ok(ValueDesc {
                dims: tensor.dims,
                element_type,
            })
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
