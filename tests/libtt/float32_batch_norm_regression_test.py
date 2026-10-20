"""float32 stablehlo.batch_norm_training/inference decompose into reductions
and elementwise ops: ttnn's batch_norm was off by about 1e-3 relative, and
torch_tpu's float16/bfloat16 training batch norms, which it computes in
float32, came out wrong."""

import jax
import numpy as np
import pytest
from jax._src.core import ShapedArray
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax.extend.core import Primitive
from jax.interpreters import mlir


def _attrs(epsilon, feature_index):
    return (
        ir.FloatAttr.get(ir.F32Type.get(), epsilon),
        ir.IntegerAttr.get(ir.IntegerType.get_signless(64), feature_index),
    )


bn_training_p = Primitive("libtt_batch_norm_training")
bn_training_p.multiple_results = True
bn_training_p.def_abstract_eval(
    lambda x, scale, offset, *, epsilon, feature_index: (
        ShapedArray(x.shape, x.dtype),
        ShapedArray(scale.shape, x.dtype),
        ShapedArray(scale.shape, x.dtype),
    )
)
mlir.register_lowering(
    bn_training_p,
    lambda ctx, x, scale, offset, *, epsilon, feature_index: hlo.BatchNormTrainingOp(
        x, scale, offset, *_attrs(epsilon, feature_index)
    ).results,
)

bn_inference_p = Primitive("libtt_batch_norm_inference")
bn_inference_p.def_abstract_eval(
    lambda x, scale, offset, mean, variance, *, epsilon, feature_index: ShapedArray(
        x.shape, x.dtype
    )
)
mlir.register_lowering(
    bn_inference_p,
    lambda ctx, x, scale, offset, mean, variance, *, epsilon, feature_index: [
        hlo.BatchNormInferenceOp(
            x, scale, offset, mean, variance, *_attrs(epsilon, feature_index)
        ).result
    ],
)

SHAPES = [((4, 5, 6, 7), 1), ((3, 10), 1), ((2, 8, 33), 1), ((2, 6, 7, 5), 3), ((2, 3, 4, 5, 6), 1)]
EPSILON = 1e-5


def per_channel(values, shape, feature_index):
    return values.reshape([-1 if d == feature_index else 1 for d in range(len(shape))])


def inputs(shape, feature_index):
    rng = np.random.default_rng(0)
    channels = shape[feature_index]
    x = (rng.standard_normal(shape) * 3 + 1).astype(np.float32)
    scale = rng.standard_normal(channels).astype(np.float32)
    offset = rng.standard_normal(channels).astype(np.float32)
    return x, scale, offset


@pytest.mark.parametrize("shape,feature_index", SHAPES)
def test_batch_norm_training(shape, feature_index):
    x, scale, offset = inputs(shape, feature_index)
    device = jax.devices("tt")[0]
    run = jax.jit(
        lambda *args: bn_training_p.bind(*args, epsilon=EPSILON, feature_index=feature_index)
    )
    output, mean, variance = run(*(jax.device_put(a, device) for a in (x, scale, offset)))

    axes = tuple(d for d in range(len(shape)) if d != feature_index)
    x64 = x.astype(np.float64)
    expected_mean = x64.mean(axis=axes)
    expected_variance = x64.var(axis=axes)
    normalized = (x64 - per_channel(expected_mean, shape, feature_index)) / np.sqrt(
        per_channel(expected_variance, shape, feature_index) + EPSILON
    )
    expected = normalized * per_channel(scale, shape, feature_index) + per_channel(
        offset, shape, feature_index
    )
    np.testing.assert_allclose(np.asarray(mean), expected_mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(variance), expected_variance, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(output), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape,feature_index", SHAPES)
def test_batch_norm_inference(shape, feature_index):
    x, scale, offset = inputs(shape, feature_index)
    rng = np.random.default_rng(1)
    mean = rng.standard_normal(shape[feature_index]).astype(np.float32)
    variance = (rng.random(shape[feature_index]) + 0.5).astype(np.float32)
    device = jax.devices("tt")[0]
    run = jax.jit(
        lambda *args: bn_inference_p.bind(*args, epsilon=EPSILON, feature_index=feature_index)
    )
    got = run(*(jax.device_put(a, device) for a in (x, scale, offset, mean, variance)))

    normalized = (x.astype(np.float64) - per_channel(mean, shape, feature_index)) / np.sqrt(
        per_channel(variance.astype(np.float64), shape, feature_index) + EPSILON
    )
    expected = normalized * per_channel(scale, shape, feature_index) + per_channel(
        offset, shape, feature_index
    )
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-5, atol=1e-5)
