#pragma once

#include "stablehlo/dialect/StablehloOps.h"

namespace libtt::mlir_frontend {

inline bool isIdentityCustomCall(mlir::stablehlo::CustomCallOp custom_call_op) {
    if (!custom_call_op || custom_call_op->getNumResults() != 1 ||
        custom_call_op.getHasSideEffect()) {
        return false;
    }
    auto call_target = custom_call_op.getCallTargetName();
    if (call_target != "annotate_device_placement" && call_target != "Sharding") {
        return false;
    }
    auto inputs = custom_call_op.getInputs();
    return inputs.size() == 1 &&
           inputs.front().getType() == custom_call_op.getResult(0).getType();
}

inline mlir::Value peelIdentityCustomCalls(mlir::Value value) {
    while (auto custom_call_op =
               value.getDefiningOp<mlir::stablehlo::CustomCallOp>()) {
        if (!isIdentityCustomCall(custom_call_op)) {
            break;
        }
        value = custom_call_op.getInputs().front();
    }
    return value;
}

template <typename OpTy>
OpTy definingOpSkippingIdentityCustomCalls(mlir::Value value) {
    return peelIdentityCustomCalls(value).template getDefiningOp<OpTy>();
}

} // namespace libtt::mlir_frontend
