#ifndef LIBTT_CPP_PJRT_INTERNAL_H_
#define LIBTT_CPP_PJRT_INTERNAL_H_

#include "xla/pjrt/c/pjrt_c_api.h"

#include <string>
#include <utility>

struct PJRT_Error {
  PJRT_Error_Code code;
  std::string message;
};

inline PJRT_Error* MakePjrtError(PJRT_Error_Code code, std::string message) {
  return new PJRT_Error{code, std::move(message)};
}

inline PJRT_Error* InvalidArgument(std::string message) {
  return MakePjrtError(PJRT_Error_Code_INVALID_ARGUMENT, std::move(message));
}

inline PJRT_Error* Unimplemented(std::string message) {
  return MakePjrtError(PJRT_Error_Code_UNIMPLEMENTED, std::move(message));
}

inline PJRT_Error* FailedPrecondition(std::string message) {
  return MakePjrtError(PJRT_Error_Code_FAILED_PRECONDITION, std::move(message));
}

inline PJRT_Error* ResourceExhausted(std::string message) {
  return MakePjrtError(PJRT_Error_Code_RESOURCE_EXHAUSTED, std::move(message));
}

inline PJRT_Error* Internal(std::string message) {
  return MakePjrtError(PJRT_Error_Code_INTERNAL, std::move(message));
}

#endif  // LIBTT_CPP_PJRT_INTERNAL_H_
