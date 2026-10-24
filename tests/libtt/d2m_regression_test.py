"""Exact small-tensor arithmetic and search loops through D2M fusion."""

import operator

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax


def _run(fn, args, path, d2m=True, sharding=None, level="O1"):
    options = {
        "optimization_level": level,
        "enable_trace": "true",
        "enable_create_d2m_subgraphs": str(d2m).lower(),
        "export_path": str(path),
    }
    device = sharding if sharding is not None else jax.devices("tt")[0]
    result = jax.jit(fn, compiler_options=options)(
        *(jax.device_put(x, device) for x in args)
    )
    result = jax.tree.map(np.asarray, result)
    (ir,) = (path / "irs").glob("ttnn_runtime_*.mlir")
    return result, ir.read_text()


@pytest.mark.parametrize(
    "compare",
    [operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge],
)
@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.float32])
def test_comparisons_preserve_operand_precision(compare, dtype, tmp_path):
    if dtype == np.int32:
        # These include float NaN/-0 bit patterns and integers that round to
        # the same bf16 value. Comparisons must use signed integer ordering.
        x = np.array(
            [-2147483648, -2147483647, -1, 0, 1, 16777217, 2139095041, 2147483647],
            dtype=dtype,
        )[None, :]
        y = np.array([0, -2147483648, 0, 0, 1, 16777216, 2139095040, -1], dtype=dtype)[
            None, :
        ]
    elif dtype == np.uint32:
        x = np.array(
            [
                0,
                1,
                2147483647,
                2147483648,
                2147483649,
                2139095041,
                4294967294,
                4294967295,
            ],
            dtype=dtype,
        )[None, :]
        y = np.array(
            [
                4294967295,
                0,
                2147483648,
                2147483647,
                2147483648,
                2139095040,
                4294967295,
                0,
            ],
            dtype=dtype,
        )[None, :]
    else:
        x = np.array(
            [0.94996, 0.95091, -0.94996, -0.95091, 1.00001, -1.00001, 0, -0.0],
            dtype=dtype,
        )[None, :]
        y = np.array([0.95, 0.95, -0.95, -0.95, 1, -1, 0, 0], dtype=dtype)[None, :]

    def select(a, b):
        a = a ^ dtype(1) if dtype != np.float32 else a + a
        return jnp.where(compare(a, b), jnp.ones_like(a), jnp.zeros_like(a))

    encoded = x ^ dtype(1) if dtype != np.float32 else x / 2
    got, ir = _run(select, (encoded, y), tmp_path)
    np.testing.assert_array_equal(got, compare(x, y).astype(dtype))
    assert '"ttnn.generic"' in ir


def test_float_predicate_selects_integer_bits(tmp_path):
    x = np.array([[0.94996, 0.95091, -0.94996, -0.95091]], np.float32)
    p = np.array([[0.95, 0.95, -0.95, -0.95]], np.float32)

    def select(a, limit):
        bits = jnp.where(a + a < limit, jnp.int32(-2147483648), jnp.int32(0))
        return bits ^ jnp.where(bits < 0, jnp.int32(0x7FFFFFFF), jnp.int32(0))

    got, ir = _run(select, (x / 2, p), tmp_path)
    np.testing.assert_array_equal(got, np.where(x < p, np.int32(-1), np.int32(0)))
    assert '"ttnn.generic"' in ir


@pytest.mark.parametrize("d2m", [False, True])
def test_boolean_where_in_mixed_precision_kernel(d2m, tmp_path):
    x = np.array([[0.1, 0.9, 0.1, 0.9]], np.float32)
    bits = np.array([[0, 0, -1, -1]], np.int32)
    finite = np.array([[True, True, False, False]])

    def select(a, b, condition):
        selected = jnp.where(condition, a + a < 1.0, b >= 0)
        return selected, b ^ jnp.where(selected, jnp.int32(-2147483648), jnp.int32(0))

    got, _ = _run(select, (x, bits, finite), tmp_path, d2m=d2m)
    expected = np.where(finite, x * 2 < 1.0, bits >= 0)
    np.testing.assert_array_equal(got[0], expected)
    np.testing.assert_array_equal(
        got[1], bits ^ np.where(expected, np.int32(-2147483648), np.int32(0))
    )


def test_integer_broadcast_arithmetic_is_exact(tmp_path):
    rng = np.random.default_rng(3)
    args = tuple(
        rng.integers(0, 1024, shape, dtype=np.int32)
        for shape in [(2, 32), (1, 32), (2, 1), (1, 1)]
    )

    def calculate(a, b, c, d):
        return (a + b) * c - d

    got, ir = _run(calculate, args, tmp_path)
    np.testing.assert_array_equal(got, calculate(*args))
    assert '"ttnn.generic"' in ir


@pytest.mark.parametrize("level", ["O0", "O1"])
@pytest.mark.parametrize("dtype", [np.int32, np.float32])
def test_shared_inputs_and_escaping_results(dtype, level, tmp_path):
    rng = np.random.default_rng(7)
    args = tuple(rng.integers(-100, 100, (2, 32)).astype(dtype) for _ in range(3))

    def calculate(a, b, c):
        shared = a + b
        return shared, shared * c - a, shared > c

    got, ir = _run(calculate, args, tmp_path, level=level)
    for actual, expected in zip(got, calculate(*args)):
        np.testing.assert_array_equal(actual, expected)
    assert ir.count('"ttnn.generic"') == 1


@pytest.mark.parametrize("select", ["minimum", "maximum"])
def test_unsigned_minmax_preserves_high_bit_ordering(select, tmp_path):
    x = np.array([[0, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF]], np.uint32)
    y = np.array([[0xFFFFFFFF, 0x80000000, 1, 0]], np.uint32)

    def calculate(a, b):
        return getattr(jnp, select)(a ^ np.uint32(1), b)

    got, _ = _run(calculate, (x, y), tmp_path)
    np.testing.assert_array_equal(got, getattr(np, select)(x ^ np.uint32(1), y))


def test_escaping_typecast_result_is_computed_once(tmp_path):
    x = np.array([[-3, 0, 1, 123, 65537, -65537]], np.int32)

    def calculate(a):
        value = a.astype(jnp.float32)
        return value, value + value

    got, _ = _run(calculate, (x,), tmp_path)
    np.testing.assert_array_equal(got[0], x.astype(np.float32))
    np.testing.assert_array_equal(got[1], x.astype(np.float32) * 2)


@pytest.mark.parametrize("shape", [(), (1,), (1, 1)])
def test_scalar_broadcast_and_escaping_counter(shape, tmp_path):
    args = (
        np.array(7, np.int32).reshape(shape),
        np.array(3, np.int32).reshape(shape),
        np.array([0x10000001, 0x7FFFFFFF], np.int32),
    )

    def calculate(counter, shift, bits):
        next_counter = counter + np.int32(1)
        bit = next_counter << shift
        return next_counter, bit ^ bits

    got, ir = _run(calculate, args, tmp_path)
    for actual, expected in zip(got, calculate(*args)):
        np.testing.assert_array_equal(actual, expected)
    assert '"ttnn.generic"' in ir


def test_scalar_shift_broadcast_preserves_live_registers(tmp_path):
    i = np.array(0, np.int32)
    bits = np.array([0, 7], np.int32)
    predicate = np.array([True, False])

    def step(i, bits, predicate):
        shift = 30 - i
        bit = jnp.int32(1) << shift
        mid = bits | bit
        selected = jnp.where(predicate, jnp.int32(0), bit)
        return shift, bit, mid, bits | selected

    got, _ = _run(step, (i, bits, predicate), tmp_path)
    bit = np.int32(1 << 30)
    expected = (
        np.int32(30),
        bit,
        bits | bit,
        bits | np.where(predicate, np.int32(0), bit),
    )
    for actual, reference in zip(got, expected):
        np.testing.assert_array_equal(actual, reference)


def test_replicated_row_and_column_broadcasts(tmp_path):
    devices = jax.devices("tt")
    if len(devices) < 2:
        pytest.skip("requires multiple TT devices")
    mesh = jax.sharding.Mesh(np.array(devices), ("devices",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    a = np.arange(4, dtype=np.int32)[None, :]
    b = np.array([[0], [3]], np.int32)
    x = np.arange(8, dtype=np.int32).reshape(2, 4) + 100
    y = -x
    got, _ = _run(
        lambda a, b, x, y: jnp.where(a == b, x, y),
        (a, b, x, y),
        tmp_path,
        sharding=sharding,
    )
    np.testing.assert_array_equal(got, np.where(a == b, x, y))


def test_fusion_does_not_cross_a_reduction_dependency(tmp_path):
    rng = np.random.default_rng(8)
    args = tuple(rng.integers(-10, 10, (2, 32)).astype(np.float32) for _ in range(2))

    def calculate(a, b):
        shared = (a + b) * b
        total = jnp.sum(shared, axis=-1, keepdims=True)
        return shared, (shared + total) * a

    got, ir = _run(calculate, args, tmp_path)
    a, b = args
    shared = (a + b) * b
    expected = (shared, (shared + shared.sum(-1, keepdims=True)) * a)
    for actual, reference in zip(got, expected):
        np.testing.assert_array_equal(actual, reference)
    assert '"ttnn.generic"' in ir


def _float_search(shape, predicate):
    """Search the signed monotonic encoding of all finite float32 values."""

    def float_bits(bits):
        return bits ^ jnp.where(bits < 0, jnp.int32(0x7FFFFFFF), jnp.int32(0))

    def test(bits):
        bits = float_bits(bits)
        finite = (bits & jnp.int32(0x7F800000)) != jnp.int32(0x7F800000)
        return jnp.where(
            finite, predicate(lax.bitcast_convert_type(bits, jnp.float32)), bits >= 0
        )

    zero = jnp.zeros(shape, jnp.int32)
    initial = jnp.where(test(zero), jnp.int32(-2147483648), zero)

    def step(i, bits):
        bit = jnp.int32(1) << (30 - i)
        return bits | jnp.where(test(bits | bit), jnp.int32(0), bit)

    bits = lax.fori_loop(0, 31, step, initial)
    return lax.bitcast_convert_type(float_bits(bits), jnp.float32)


@pytest.mark.parametrize("mode", ["topp", "topk", "topk_negative"])
def test_search_matches_unfused_and_numpy(mode, tmp_path):
    x = np.random.default_rng(0).standard_normal((2, 2048), dtype=np.float32) * 4
    if mode == "topp":
        x = np.exp(x - x.max(-1, keepdims=True))
        x /= x.sum(-1, keepdims=True)
        limit = np.array([0.95, 0.9], np.float32)

        def search(x, p):
            xt = jnp.swapaxes(x, -1, -2)
            return _float_search(
                x.shape[:-1],
                lambda t: jnp.sum(jnp.where(xt >= t[None, :], xt, 0), axis=0) < p,
            )

    else:
        if mode == "topk_negative":
            x -= 20
        limit = np.array([20, 50], np.int32)

        def search(x, k):
            xt = jnp.swapaxes(x, -1, -2)
            return -_float_search(
                x.shape[:-1], lambda t: jnp.sum(xt > (-t)[None, :], axis=0) >= k
            )

    reference, _ = _run(search, (x, limit), tmp_path / "plain", d2m=False)
    got, ir = _run(search, (x, limit), tmp_path / "d2m")
    np.testing.assert_array_equal(got.view(np.uint32), reference.view(np.uint32))
    if mode == "topp":
        mass = np.where(x >= got[:, None], x, 0).sum(-1)
        next_mass = np.where(x >= np.nextafter(got, np.inf)[:, None], x, 0).sum(-1)
        assert np.all(mass >= limit)
        assert np.all(next_mass < limit)
    else:
        np.testing.assert_array_equal((x >= got[:, None]).sum(-1), limit)
    assert '"ttnn.generic"' in ir
    assert '"ttnn.masked_reduce"' in ir
    assert '"ttnn.execute_trace"' in ir
