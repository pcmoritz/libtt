use crate::dram::DType;
use crate::PJRT_Buffer_Type;
use std::io;

pub(crate) fn pjrt_buffer_type_to_dtype(buffer_type: PJRT_Buffer_Type) -> io::Result<DType> {
    match buffer_type {
        PJRT_Buffer_Type::PJRT_Buffer_Type_S8 => Ok(DType::Int8),
        PJRT_Buffer_Type::PJRT_Buffer_Type_PRED => Ok(DType::UInt8),
        PJRT_Buffer_Type::PJRT_Buffer_Type_S32 => Ok(DType::Int32),
        PJRT_Buffer_Type::PJRT_Buffer_Type_U8 => Ok(DType::UInt8),
        PJRT_Buffer_Type::PJRT_Buffer_Type_U16 => Ok(DType::UInt16),
        PJRT_Buffer_Type::PJRT_Buffer_Type_U32 => Ok(DType::UInt32),
        PJRT_Buffer_Type::PJRT_Buffer_Type_F16 => Ok(DType::Float16),
        PJRT_Buffer_Type::PJRT_Buffer_Type_F32 => Ok(DType::Float32),
        PJRT_Buffer_Type::PJRT_Buffer_Type_BF16 => Ok(DType::Float16B),
        PJRT_Buffer_Type::PJRT_Buffer_Type_INVALID => {
            Err(invalid_input("invalid PJRT buffer type"))
        }
        other => Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!("unsupported PJRT buffer type {other:?}"),
        )),
    }
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}
