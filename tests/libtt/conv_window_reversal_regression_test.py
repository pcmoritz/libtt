"""torch_tpu convolutions express kernel reversal through window_reversal."""

import jax
import numpy as np
import pytest
from jax._src.core import ShapedArray
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax.extend.core import Primitive
from jax.interpreters import mlir

conv_p = Primitive("libtt_conv_window_reversal")


def _abstract(x, w, *, stride, padding, reversal):
    out = [
        (size - 1) * stride + low + high - k + 2
        for size, (low, high), k in zip(x.shape[2:], padding, w.shape[2:])
    ]
    return ShapedArray((x.shape[0], w.shape[0], *out), x.dtype)


def _lower(ctx, x, w, *, stride, padding, reversal):
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
            window_reversal=None if reversal is None else ir.DenseBoolArrayAttr.get(reversal),
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


conv_p.def_abstract_eval(_abstract)
mlir.register_lowering(conv_p, _lower)


@pytest.mark.parametrize("level", ["O0", "O1"])
@pytest.mark.parametrize(
    "x_shape,w_shape,stride,padding,reversal",
    [
        ((1, 2, 5, 6), (3, 2, 3, 3), 2, ((1, 1), (1, 1)), (True, True)),
        ((2, 4, 7), (3, 4, 3), 2, ((2, 2),), (True,)),
        ((2, 4, 6, 5), (3, 4, 2, 3), 1, ((1, 1), (2, 2)), (True, False)),
        ((2, 4, 6, 5), (3, 4, 2, 3), 1, ((1, 1), (2, 2)), (False, True)),
        ((1, 2, 5, 6), (3, 2, 3, 3), 1, ((-1, 0), (0, -1)), (True, True)),
        ((1, 2, 5, 6), (3, 2, 3, 3), 1, ((1, 1), (1, 1)), (False, False)),
        ((1, 2, 5, 6), (3, 2, 3, 3), 1, ((1, 1), (1, 1)), None),
    ],
    ids=["transpose-2d", "transpose-1d", "reverse-height", "reverse-width", "negative-padding", "no-reversal", "no-attribute"],
)
def test_conv_window_reversal(x_shape, w_shape, stride, padding, reversal, level):
    # Small integer-valued floats make the orientation check exact even with
    # the device convolution's reduced floating-point precision.
    rng = np.random.default_rng(0)
    x = rng.integers(-2, 3, x_shape).astype(np.float32)
    w = rng.integers(-2, 3, w_shape).astype(np.float32)

    def run(x, w):
        return conv_p.bind(x, w, stride=stride, padding=padding, reversal=reversal)

    device = jax.devices("tt")[0]
    got = np.asarray(jax.jit(run, compiler_options={"optimization_level": level})(
        jax.device_put(x, device), jax.device_put(w, device)))
    expected = np.asarray(jax.jit(run, backend="cpu")(x, w))
    np.testing.assert_array_equal(got, expected)
