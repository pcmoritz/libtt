"""float32 stablehlo.batch_norm_training/inference against float64.

ttnn's batch_norm normalizes exactly, but ran with approximate math (off by
about 1e-3) when the program's compute config left it unset, and training
returned the running statistics, with Bessel's correction, instead of
StableHLO's biased batch variance. Training now returns the batch mean and
variance that ttnn::batch_norm_training normalizes with."""

import re

import jax
import jax.numpy as jnp
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

SHAPES = [
    ((4, 5, 6, 7), 1),
    ((3, 10), 1),
    ((2, 8, 33), 1),
    ((2, 6, 7, 5), 3),
    ((2, 3, 4, 5, 6), 1),
    ((8, 3, 56, 56), 1),  # 25088 elements per channel
]
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


@pytest.mark.parametrize("optimization_level", ["O0", "O1"])
@pytest.mark.parametrize("shape,feature_index", SHAPES)
def test_batch_norm_training(shape, feature_index, optimization_level, tmp_path):
    x, scale, offset = inputs(shape, feature_index)
    device = jax.devices("tt")[0]
    run = jax.jit(
        lambda *args: bn_training_p.bind(*args, epsilon=EPSILON, feature_index=feature_index),
        compiler_options={"optimization_level": optimization_level, "export_path": str(tmp_path)},
    )
    output, mean, variance = run(*(jax.device_put(a, device) for a in (x, scale, offset)))
    # One native op returns the output and the batch statistics.
    (ir,) = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    assert len(re.findall(r'%\w+, %\w+, %\w+ = "ttnn\.batch_norm_training"', ir)) == 1
    assert "ttnn.mean" not in ir

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


def run_training(x, scale, offset, epsilon, feature_index):
    device = jax.devices("tt")[0]
    run = jax.jit(
        lambda *args: bn_training_p.bind(*args, epsilon=epsilon, feature_index=feature_index)
    )
    return [np.asarray(v) for v in run(*(jax.device_put(a, device) for a in (x, scale, offset)))]


def test_batch_norm_training_one_element_per_channel():
    """Each channel reduces one element: the variance is 0 and the output is the
    offset, with nothing divided by zero."""
    x, scale, offset = inputs((1, 3), 1)
    output, mean, variance = run_training(x, scale, offset, EPSILON, 1)
    for value in (output, mean, variance):
        assert np.all(np.isfinite(value))
    np.testing.assert_array_equal(mean, x[0])
    np.testing.assert_array_equal(variance, np.zeros(3, np.float32))
    np.testing.assert_allclose(output[0], offset, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("training", [True, False])
def test_batch_norm_small_epsilon(training):
    shape, epsilon = (4, 5, 6, 7), 1e-12
    x, scale, offset = inputs(shape, 1)
    x = x * np.float32(1e-3)
    x64 = x.astype(np.float64)
    mean, variance = x64.mean(axis=(0, 2, 3)), x64.var(axis=(0, 2, 3))
    if training:
        output = run_training(x, scale, offset, epsilon, 1)[0]
    else:
        device = jax.devices("tt")[0]
        run = jax.jit(lambda *args: bn_inference_p.bind(*args, epsilon=epsilon, feature_index=1))
        mean, variance = mean.astype(np.float32), variance.astype(np.float32)
        output = np.asarray(run(*(jax.device_put(a, device) for a in (x, scale, offset, mean, variance))))
    normalized = (x64 - per_channel(mean, shape, 1)) / np.sqrt(per_channel(variance, shape, 1) + epsilon)
    expected = normalized * per_channel(scale, shape, 1) + per_channel(offset, shape, 1)
    assert np.all(np.isfinite(output))
    np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-5)


def test_batch_norm_training_high_offset():
    """x = 1000 + 0.01 * noise: float32 resolves x to 6e-5, so the mean and the
    output are only that accurate, while the centered variance stays close."""
    rng = np.random.default_rng(2)
    shape = (4, 5, 6, 7)
    x = (1000 + 0.01 * rng.standard_normal(shape)).astype(np.float32)
    scale, offset = np.ones(5, np.float32), np.zeros(5, np.float32)
    output, mean, variance = run_training(x, scale, offset, EPSILON, 1)
    x64 = x.astype(np.float64)
    for value in (output, mean, variance):
        assert np.all(np.isfinite(value))
    mean64, variance64 = x64.mean(axis=(0, 2, 3)), x64.var(axis=(0, 2, 3))
    np.testing.assert_allclose(mean, mean64, rtol=1e-7)
    np.testing.assert_allclose(variance, variance64, rtol=1e-3)
    expected = (x64 - per_channel(mean64, shape, 1)) / np.sqrt(per_channel(variance64, shape, 1) + EPSILON)
    np.testing.assert_allclose(output, expected, atol=1e-2)


def nchw_training_reference(x64):
    mean, variance = x64.mean(axis=(0, 2, 3)), x64.var(axis=(0, 2, 3))
    output = (x64 - mean[None, :, None, None]) / np.sqrt(variance[None, :, None, None] + EPSILON)
    return output, mean, variance


@pytest.mark.parametrize("optimization_level", ["O0", "O1"])
def test_batch_norm_training_constant_scale_and_offset(optimization_level):
    """Scale 1 and offset 0 as constants, like the zero and one running
    statistics that training used to update in place, so that later calls and
    later batch norms read updated constants: two batch norms in one graph,
    run on new inputs three times."""

    def two_batch_norms(x):
        ones, zeros = jnp.ones(x.shape[1], x.dtype), jnp.zeros(x.shape[1], x.dtype)
        first = bn_training_p.bind(x, ones, zeros, epsilon=EPSILON, feature_index=1)
        second = bn_training_p.bind(first[0] * 2 + 3, ones, zeros, epsilon=EPSILON, feature_index=1)
        return *first, *second

    run = jax.jit(two_batch_norms, compiler_options={"optimization_level": optimization_level})
    device = jax.devices("tt")[0]
    for call in range(3):
        x = (np.random.default_rng(call).standard_normal((4, 5, 6, 7)) * (call + 1) + call).astype(np.float32)
        first = nchw_training_reference(x.astype(np.float64))
        expected = first + nchw_training_reference(first[0] * 2 + 3)
        got = run(jax.device_put(x, device))
        for value, want in zip(got, expected):
            np.testing.assert_allclose(np.asarray(value), want, rtol=1e-5, atol=1e-5)
