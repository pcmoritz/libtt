"""Reductions over the rows of a tall, thin tensor run as wide-row reductions.

The binary searches behind SGLang-JAX's top-k and top-p masks swap the vocab
axis to the front and reduce over it 32 times per mask. On TT a [vocab, batch]
tensor has one tile column, so that reduction, and even the elementwise ops
feeding it, run on one core: 2.9 ms per search step against 0.5 ms on
[batch, vocab]. The compiler rebuilds such a reduction chain on the transposed
data in full tiles, [batch, vocab / C, C], and reduces the last two dimensions
(tt_mlir_wide_reductions.patch); C is the widest 32 * 2^k that divides the
vocab up to about sqrt(vocab). Vocabs that do not split are left alone.

Wide softmax rows are the other sampler cost: TTNN's softmax parallelizes over
rows, so two rows of a vocabulary take 12 ms where the decomposition takes 1 ms.
The softmax fusion skips that shape.

The rewrite runs at the end of the StableHLO pipeline, on device-local shapes,
so the program's sharding is already decided and stays as it is.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax._src.lib.mlir import ir
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

# Vocab sizes and the chunked shape the search reduces: 2048 splits into 32
# columns (64 > sqrt(2048)), 16384 into 128, and 2000 does not split (the
# search runs as written).
WIDTHS = {2048: "2x64x32", 16384: "2x128x128", 2000: None}


def _source(width):
    return np.random.default_rng(0).standard_normal((2, width), dtype=np.float32) * 4


def _jit(fn, trace, export_path=None):
    options = {"optimization_level": "O1", "enable_trace": str(trace).lower()}
    if export_path is not None:
        options["export_path"] = str(export_path)
    return jax.jit(fn, compiler_options=options)


def _put(value):
    return jax.device_put(np.asarray(value), jax.devices("tt")[0])


def _runtime_ir(export_path):
    (ir,) = (export_path / "irs").glob("ttnn_runtime_*.mlir")
    return ir.read_text()


def _count_threshold(x, k):
    """The largest threshold with at least k values above it in each row, found
    the way SGLang-JAX's masks search: 31 reductions over the vocab axis, which
    the search puts first. Thresholds stay positive here (the k-th largest of
    these rows is), so float bit patterns order like integers."""
    xt = jnp.swapaxes(x, 0, 1)

    def body(i, bits):
        bit = jnp.int32(1 << (30 - i))
        candidate = bits | bit
        threshold = lax.bitcast_convert_type(candidate, jnp.float32)
        count = jnp.sum(xt > threshold[None, :], axis=0)
        return bits | jnp.where(count >= k, bit, jnp.int32(0))

    bits = lax.fori_loop(0, 31, body, jnp.zeros((x.shape[0],), jnp.int32))
    return lax.bitcast_convert_type(bits, jnp.float32)


def _search_sums(ir):
    # The input shapes of the program's sums, such as "2x64x32".
    shapes = []
    for line in ir.splitlines():
        if '"ttnn.sum"' in line:
            match = re.search(r"\(tensor<((?:\d+x)+)", line)
            shapes.append(match.group(1).rstrip("x"))
    return shapes


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("trace", [False, True])
def test_search_reduces_over_wide_rows(trace, width, tmp_path):
    k = 20
    source = _source(width)
    run = _jit(lambda x: _count_threshold(x, k), trace, tmp_path)
    threshold = np.asarray(run(_put(source)))
    # At least k values lie above the threshold, and fewer above the next
    # larger float.
    assert np.all((source > threshold[:, None]).sum(axis=1) >= k)
    assert np.all((source > np.nextafter(threshold, np.inf)[:, None]).sum(axis=1) < k)
    chunked = WIDTHS[width]
    if chunked is None:
        return
    ir = _runtime_ir(tmp_path)
    # No transpose survives, and the search sums full tiles.
    assert "ttnn.permute" not in ir and "ttnn.transpose" not in ir
    assert re.search(r'"ttnn\.sum".*dim_arg = \[0 : i32\]', ir) is None
    assert chunked in _search_sums(ir)


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("trace", [False, True])
def test_mass_search_reduces_over_wide_rows(trace, width):
    # The top-p variant: a masked sum of probabilities over the vocab axis.
    # The device's softmax is the reference, since the search ran on it.
    p = 0.9

    def mass_threshold(x):
        probs = jax.nn.softmax(x, axis=-1)
        pt = jnp.swapaxes(probs, 0, 1)

        def body(i, bits):
            bit = jnp.int32(1 << (30 - i))
            candidate = bits | bit
            threshold = lax.bitcast_convert_type(candidate, jnp.float32)
            mass = jnp.sum(jnp.where(pt >= threshold[None, :], pt, 0.0), axis=0)
            return bits | jnp.where(mass >= p, bit, jnp.int32(0))

        bits = lax.fori_loop(0, 31, body, jnp.zeros((x.shape[0],), jnp.int32))
        return lax.bitcast_convert_type(bits, jnp.float32), probs

    run = _jit(mass_threshold, trace)
    threshold, probs = (np.asarray(v) for v in run(_put(_source(width))))
    kept = np.where(probs >= threshold[:, None], probs, 0.0).sum(axis=1)
    assert np.all(kept >= p - 1e-3)
    dropped = np.where(
        probs >= np.nextafter(threshold, np.inf)[:, None], probs, 0.0
    ).sum(axis=1)
    assert np.all(dropped < p + 1e-3)


def _mesh(rows):
    devices = jax.devices("tt")
    if len(devices) < 2 or rows % len(devices):
        pytest.skip(f"needs a mesh that splits {rows} rows")
    return Mesh(np.array(devices), ("x",))


def _walk(op):
    for region in op.regions:
        for block in region.blocks:
            for inner in block.operations:
                yield inner
                yield from _walk(inner)


@pytest.mark.parametrize("spec", [P(), P("x", None)], ids=["replicated", "batch"])
def test_search_on_a_mesh(spec, tmp_path):
    # The sampler's logits are replicated under tensor parallelism, and their
    # search is rewritten on each device. A batch-sharded source reaches the
    # loop gathered and is searched as written.
    k = 20
    source = np.random.default_rng(2).standard_normal((8, 2048), dtype=np.float32) * 4
    x = jax.device_put(source, NamedSharding(_mesh(8), spec))
    threshold = np.asarray(_jit(lambda v: _count_threshold(v, k), False, tmp_path)(x))
    assert np.all((source > threshold[:, None]).sum(axis=1) >= k)
    assert np.all((source > np.nextafter(threshold, np.inf)[:, None]).sum(axis=1) < k)
    if spec == P():
        assert "8x64x32" in _search_sums(_runtime_ir(tmp_path))


def test_reduce_keeps_its_own_sharding(tmp_path):
    # Imported XLA programs can pin a reduce's result. Pinned to "x", the
    # rewritten reduce runs on each device's rows.
    source = (np.arange(8 * 2048, dtype=np.float32).reshape(8, 2048) % 5) - 2
    mesh = _mesh(8)
    x = jax.device_put(source, NamedSharding(mesh, P()))
    run = _jit(lambda v: jnp.sum(jnp.swapaxes(v, 0, 1) * 2.0, axis=0), False, tmp_path)
    lowered = run.lower(x)
    module = lowered.compiler_ir("stablehlo")
    (reduce,) = [op for op in _walk(module.operation) if op.name == "stablehlo.reduce"]
    with module.context:
        reduce.attributes["sdy.sharding"] = ir.Attribute.parse(
            '#sdy.sharding_per_value<[<@mesh, [{"x"}]>]>'
        )
    result = np.asarray(lowered.compile()(x))
    np.testing.assert_array_equal(result, (source.T * 2.0).sum(axis=0))
    assert _search_sums(_runtime_ir(tmp_path)) == [f"{8 // mesh.size}x64x32"]


@pytest.mark.parametrize("trace", [False, True])
def test_wide_softmax_stays_decomposed(trace, tmp_path):
    x = np.random.default_rng(1).standard_normal((2, 16384), dtype=np.float32) * 3
    run = _jit(lambda v: jax.nn.softmax(v, axis=-1), trace, tmp_path)
    result = np.asarray(run(_put(x)))
    expected = np.asarray(jax.nn.softmax(jnp.asarray(x), axis=-1))
    np.testing.assert_allclose(result, expected, rtol=2e-3, atol=1e-6)
    assert "ttnn.softmax" not in _runtime_ir(tmp_path)
