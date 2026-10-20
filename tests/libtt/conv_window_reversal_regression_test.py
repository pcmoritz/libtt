"""A stablehlo.convolution with window_reversal, as torch_tpu writes its
transposed convolutions (conv_transpose1d/2d), failed to legalize; it now
reverses the kernel explicitly, as JAX's transposed convolutions do."""

import jax
import numpy as np
import pytest
from jax._src.core import ShapedArray
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax.extend.core import Primitive
from jax.interpreters import mlir

conv_transpose_p = Primitive("libtt_conv_window_reversal")


def _abstract(x, w, *, stride, padding):
    n, _, *spatial = x.shape
    out = [
        (size - 1) * stride + 1 + low + high - k + 1
        for size, (low, high), k in zip(spatial, padding, w.shape[2:])
    ]
    return ShapedArray((n, w.shape[0], *out), x.dtype)


def _lower(ctx, x, w, *, stride, padding):
    rank = len(padding)
    spatial = list(range(2, 2 + rank))
    i64 = ir.IntegerType.get_signless(64)
    aval = ctx.avals_out[0]
    return [
        hlo.ConvolutionOp(
            ir.RankedTensorType.get(aval.shape, mlir.dtype_to_ir_type(aval.dtype)),
            x,
            w,
            padding=ir.DenseIntElementsAttr.get(np.array(padding, np.int64), type=i64),
            lhs_dilation=ir.DenseI64ArrayAttr.get([stride] * rank),
            window_reversal=ir.DenseBoolArrayAttr.get([True] * rank),
            dimension_numbers=hlo.ConvDimensionNumbers.get(
                input_batch_dimension=0,
                input_feature_dimension=1,
                input_spatial_dimensions=spatial,
                kernel_output_feature_dimension=0,
                kernel_input_feature_dimension=1,
                kernel_spatial_dimensions=spatial,
                output_batch_dimension=0,
                output_feature_dimension=1,
                output_spatial_dimensions=spatial,
            ),
            feature_group_count=ir.IntegerAttr.get(i64, 1),
            batch_group_count=ir.IntegerAttr.get(i64, 1),
        ).result
    ]


conv_transpose_p.def_abstract_eval(_abstract)
mlir.register_lowering(conv_transpose_p, _lower)


@pytest.mark.parametrize(
    "x_shape,w_shape,stride,padding",
    [
        ((1, 2, 5, 6), (3, 2, 3, 3), 2, ((1, 1), (1, 1))),
        ((2, 4, 6, 5), (3, 4, 2, 3), 1, ((1, 1), (2, 2))),
        ((2, 4, 7), (3, 4, 3), 2, ((2, 2),)),
    ],
)
def test_conv_window_reversal(x_shape, w_shape, stride, padding):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(x_shape).astype(np.float32)
    w = rng.standard_normal(w_shape).astype(np.float32)

    def run(x, w):
        return conv_transpose_p.bind(x, w, stride=stride, padding=padding)

    device = jax.devices("tt")[0]
    got = np.asarray(jax.jit(run)(jax.device_put(x, device), jax.device_put(w, device)))
    expected = np.asarray(jax.jit(run, backend="cpu")(x, w))
    # Device convolutions are about 1.5e-3 off relative to the largest
    # magnitude; an unreversed kernel is off by order 1.
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-2 * np.abs(expected).max())
