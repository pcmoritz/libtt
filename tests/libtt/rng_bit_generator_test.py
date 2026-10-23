"""stablehlo.rng_bit_generator: the bits are a function of the state.

libtt computes them with Threefry-2x32 (20 rounds) for DEFAULT and THREE_FRY,
and with Philox-4x32 (10 rounds) for PHILOX. The key is (state[0], 0), and
element i (in row-major order) is the first word of counter (i, state[1]) for
Threefry and (i, state[1], state[2] or 0, 0) for Philox. The output state
advances state[1] by one per nonempty call, so the next call uses other
counters.

The states are u64[2], as torch_tpu passes them. (JAX's u32[4] keys need a
width-changing bitcast to u64[2], which libtt doesn't compile.) The device's
64-bit integers hold 32 bits: the state's words count modulo 2^32, and 64-bit
outputs have 32 random low bits.
"""

import jax
from jax import core
from jax._src.lib.mlir.dialects import hlo
from jax.extend.core import Primitive
from jax.interpreters import mlir
import jax.numpy as jnp
import numpy as np
import pytest

M32 = 0xFFFFFFFF
ALGORITHMS = {
    "DEFAULT": jax.lax.RandomAlgorithm.RNG_DEFAULT,
    "THREE_FRY": jax.lax.RandomAlgorithm.RNG_THREE_FRY,
    "PHILOX": jax.lax.RandomAlgorithm.RNG_PHILOX,
}


def bits(state, shape, dtype, algorithm="DEFAULT"):
    with jax.enable_x64(True):
        new_state, values = jax.jit(
            lambda state: jax.lax.rng_bit_generator(
                state, shape, dtype, algorithm=ALGORITHMS[algorithm]
            )
        )(jnp.asarray(state, dtype=jnp.uint64))
    return np.asarray(new_state), np.asarray(values)


# stablehlo.rng_bit_generator with any state and output types, which JAX's
# lax.rng_bit_generator doesn't allow.
rng_bit_generator_p = Primitive("test_rng_bit_generator")
rng_bit_generator_p.multiple_results = True
rng_bit_generator_p.def_abstract_eval(
    lambda state, *, shape, dtype, algorithm: (
        core.ShapedArray(state.shape, state.dtype),
        core.ShapedArray(shape, dtype),
    )
)
mlir.register_lowering(
    rng_bit_generator_p,
    lambda ctx, state, *, shape, dtype, algorithm: hlo.RngBitGeneratorOp(
        *(mlir.aval_to_ir_type(ctx.module_context, aval) for aval in ctx.avals_out),
        hlo.RngAlgorithmAttr.get(algorithm),
        state,
    ).results,
)


def raw_bits(state, shape, dtype, algorithm):
    with jax.enable_x64(True):
        new_state, values = jax.jit(
            lambda state: rng_bit_generator_p.bind(
                state, shape=shape, dtype=np.dtype(dtype), algorithm=algorithm
            )
        )(jnp.asarray(state))
    return np.asarray(new_state), np.asarray(values)


def state(*words):
    return np.array(words, dtype=np.uint64)


def threefry2x32(key, counters):
    """Threefry-2x32 with 20 rounds, as in JAX."""
    ks = [np.uint32(key[0]), np.uint32(key[1])]
    ks.append(ks[0] ^ ks[1] ^ np.uint32(0x1BD11BDA))
    x0 = np.asarray(counters[0], np.uint32) + ks[0]
    x1 = np.asarray(counters[1], np.uint32) + ks[1]
    for i in range(5):
        for r in ((13, 15, 26, 6), (17, 29, 16, 24))[i % 2]:
            x0 = x0 + x1
            x1 = ((x1 << np.uint32(r)) | (x1 >> np.uint32(32 - r))) ^ x0
        x0 = x0 + ks[(i + 1) % 3]
        x1 = x1 + ks[(i + 2) % 3] + np.uint32(i + 1)
    return [x0, x1]


def philox4x32(key, counters):
    """Philox-4x32 with 10 rounds, as Random123's philox4x32_10."""
    k0, k1 = key
    x = [np.asarray(c, np.uint32) for c in counters]
    for r in range(10):
        if r:
            k0, k1 = (k0 + 0x9E3779B9) & M32, (k1 + 0xBB67AE85) & M32
        p0 = x[0].astype(np.uint64) * np.uint64(0xD2511F53)
        p1 = x[2].astype(np.uint64) * np.uint64(0xCD9E8D57)
        hi0, lo0 = (p0 >> np.uint64(32)).astype(np.uint32), p0.astype(np.uint32)
        hi1, lo1 = (p1 >> np.uint64(32)).astype(np.uint32), p1.astype(np.uint32)
        x = [hi1 ^ x[1] ^ np.uint32(k0), lo1, hi0 ^ x[3] ^ np.uint32(k1), lo0]
    return x


def expected(algorithm, state, elements):
    """The 32-bit words that libtt draws from a state, in row-major order."""
    seed, offset, *rest = (int(word) & M32 for word in state)
    i, n = np.arange(elements), elements
    if algorithm == "PHILOX":
        high = rest[0] if rest else 0
        return philox4x32((seed, 0), [i, np.full(n, offset), np.full(n, high), np.zeros(n)])[0]
    return threefry2x32((seed, 0), [i, np.full(n, offset)])[0]


def advanced(old, elements):
    new = old.copy()
    new[1] = (int(old[1]) + bool(elements)) & M32
    return new


# Random123's known-answer vectors.
@pytest.mark.parametrize(
    "generator, key, counters, words",
    [
        (threefry2x32, (0, 0), (0, 0), (0x6B200159, 0x99BA4EFE)),
        (threefry2x32, (M32, M32), (M32, M32), (0x1CB996FC, 0xBB002BE7)),
        (threefry2x32, (0x13198A2E, 0x03707344), (0x243F6A88, 0x85A308D3), (0xC4923A9C, 0x483DF7A0)),
        (philox4x32, (0, 0), (0, 0, 0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
        (philox4x32, (M32, M32), (M32,) * 4, (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
        (
            philox4x32,
            (0xA4093822, 0x299F31D0),
            (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
            (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
        ),
    ],
)
def test_references_match_known_answers(generator, key, counters, words):
    got = generator(key, [np.array([c], np.uint32) for c in counters])
    assert tuple(int(w[0]) for w in got) == words


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.uint64])
def test_bits_match_reference(algorithm, dtype):
    """8- and 16-bit elements are the words' low bits; 64-bit elements are the
    words, as the device's 64-bit integers hold 32 bits."""
    old = state(0xDEADBEEF, 3)
    new, values = bits(old, (5, 7), dtype, algorithm)
    np.testing.assert_array_equal(values, expected(algorithm, old, 35).astype(dtype).reshape(5, 7))
    assert values.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(new, advanced(old, 35))


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("shape", [(), (1,), (3, 5, 7), (2, 1, 33)])
def test_shapes(algorithm, shape):
    old = state(0xDEADBEEF, 3)
    elements = int(np.prod(shape))
    new, values = bits(old, shape, np.uint32, algorithm)
    np.testing.assert_array_equal(values, expected(algorithm, old, elements).reshape(shape))
    np.testing.assert_array_equal(new, advanced(old, elements))


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_large_output(algorithm):
    """16M elements, as torch.rand(4096, 4096) draws."""
    old = state(7, 1)
    new, values = bits(old, (4096, 4096), np.uint32, algorithm)
    np.testing.assert_array_equal(values.ravel(), expected(algorithm, old, 4096 * 4096))
    np.testing.assert_array_equal(new, advanced(old, 4096 * 4096))


def test_philox_three_word_state():
    old = state(0xDEADBEEF, 3, 5)
    new, values = raw_bits(old, (9,), np.uint32, "PHILOX")
    np.testing.assert_array_equal(values, expected("PHILOX", old, 9))
    np.testing.assert_array_equal(new, advanced(old, 9))


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_consecutive_calls_use_other_counters(algorithm):
    """Each call's state is the last call's output state, so the counters'
    second words differ, whatever the sizes."""
    current, drawn = state(11, 0), []
    for elements in (1, 3, 35, 2, 64):
        new, values = bits(current, (elements,), np.uint32, algorithm)
        np.testing.assert_array_equal(values, expected(algorithm, current, elements))
        np.testing.assert_array_equal(new, advanced(current, elements))
        current = new
        drawn.append(values)
    drawn = np.concatenate(drawn)
    assert len(np.unique(drawn)) == len(drawn)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
def test_same_state_gives_same_bits(algorithm, dtype):
    _, first = bits(state(42, 0), (64, 128), dtype, algorithm)
    _, second = bits(state(42, 0), (64, 128), dtype, algorithm)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_other_states_give_other_bits(algorithm):
    _, base = bits(state(1, 0), (32, 32), np.uint32, algorithm)
    _, other_seed = bits(state(2, 0), (32, 32), np.uint32, algorithm)
    _, other_offset = bits(state(1, 1), (32, 32), np.uint32, algorithm)
    for other in (other_seed, other_offset):
        assert np.mean(base == other) < 0.01


def test_state_words_count_modulo_2_32():
    """The device's 64-bit integers hold 32 bits, so states that differ only
    above bit 31 give the same bits."""
    _, base = bits(state(5, 1), (64,), np.uint32)
    for other in (state(5 + 2**32, 1), state(5, 1 + 2**32)):
        np.testing.assert_array_equal(bits(other, (64,), np.uint32)[1], base)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_counter_wraps(algorithm):
    old = state(5, M32)
    new, values = bits(old, (35,), np.uint32, algorithm)
    np.testing.assert_array_equal(new, state(5, 0))
    np.testing.assert_array_equal(values, expected(algorithm, old, 35))


def test_large_draw_does_not_repeat_after_1024_calls():
    # Advancing by ceil(elements / 4) used to repeat the entire tensor here.
    with jax.enable_x64(True):
        draw = jax.jit(lambda s: jax.lax.rng_bit_generator(s, (4096, 4096), jnp.uint32))
        current, first = draw(jnp.asarray(state(7, 0)))
        for _ in range(1024):
            current, last = draw(current)
        np.testing.assert_array_equal(np.asarray(current), state(7, 1025))
        assert not np.array_equal(np.asarray(first), np.asarray(last))


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("dtype, width", [(np.uint8, 8), (np.uint16, 16), (np.uint32, 32), (np.uint64, 32)])
def test_bits_are_balanced(algorithm, dtype, width):
    """Each random bit is set half the time. 64-bit elements have 32 random
    bits."""
    _, values = bits(state(3, 5), (256, 256), dtype, algorithm)
    values = values.astype(np.uint64).ravel()
    assert np.all(values >> np.uint64(width) == 0)
    ones = np.array([np.mean((values >> np.uint64(b)) & np.uint64(1)) for b in range(width)])
    # 65536 samples per bit: the standard deviation of the mean is 0.002.
    np.testing.assert_allclose(ones, 0.5, atol=0.01)
    # About as many distinct values as random draws give.
    distinct = 2.0**width * -np.expm1(values.size * np.log1p(-(2.0**-width)))
    assert len(np.unique(values)) > 0.95 * distinct


def test_empty_output_keeps_the_state():
    new, values = bits(state(1, 2), (0, 3), np.uint32)
    assert values.shape == (0, 3)
    np.testing.assert_array_equal(new, state(1, 2))


@pytest.mark.parametrize(
    "old, dtype, message",
    [
        (state(1, 2), np.float32, "generates unsigned integers of up to 64 bits, not 'f32'"),
        (state(1, 2), np.int32, "generates unsigned integers of up to 64 bits, not 'i32'"),
        (state(1), np.uint32, "needs a state of at least two 32- or 64-bit integers"),
    ],
)
def test_unsupported_types_are_errors(old, dtype, message):
    with pytest.raises(jax.errors.JaxRuntimeError, match=message):
        raw_bits(old, (4,), dtype, "DEFAULT")
