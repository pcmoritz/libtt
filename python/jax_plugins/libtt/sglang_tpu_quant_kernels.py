"""SGLang-TorchTPU's quantized Pallas kernels as plain JAX, for Tenstorrent.

SGLang-TorchTPU (`sglang_tpu`) exports JAX functions through torch_tpu, and
its quantized linear layers and fused MoE call Pallas TPU kernels that only
TPUs compile. In TPU compatibility mode, `tpu_compat` replaces each kernel
with the JAX function below when `sglang_tpu` imports its module. Each takes
the kernel's arguments and computes the kernel's math with ordinary JAX ops.
FP8 operands are converted to BF16, which holds every E4M3 value exactly,
before their dots: libtt's FP8 dot rounds more coarsely, while the BF16 dot
gives the same bits as SGLang's XLA path (torch._scaled_mm).

- `fp8_linear_small_m`: W8A8 linear for decode-sized batches. Each activation
  row is quantized to FP8 with its own scale, multiplied with the FP8 weight
  in an FP32-accumulated dot, and rescaled.
- `_blockwise_quantized_matmul_kernel`: block-FP8 W8A8 matmul. Activations are
  quantized per row and K block, each block's dot is rescaled by both scales,
  and the blocks are summed (in FP32; the kernel sums in BF16).
- `_fused_moe_jax`: fused MoE over BF16 or FP8 experts, with per-expert or
  block scales. It dequantizes the weights, runs every local expert on every
  token, and combines the outputs with the routing weights. That costs
  experts / top-k times the FLOPs of routed experts, which is fine for tests
  and small models; packed FP4 and int4 weights aren't supported.
"""

SMALL_M_MODULE = "sglang_tpu.kernels.fp8_linear_small_m"
BLOCK_FP8_MODULE = "sglang_tpu.quantized_matmul"
FUSED_MOE_MODULE = "sglang_tpu.moe.pallas.fused_moe_op"


def _fp8_linear_small_m(x, weight, weight_scale):
    """x [M, K], weight [K, N] FP8, weight_scale [N] or scalar."""
    import jax
    import jax.numpy as jnp

    fp8_max = float(jnp.finfo(weight.dtype).max)
    xf = x.astype(jnp.float32)
    x_scale = jnp.maximum(jnp.max(jnp.abs(xf), axis=-1, keepdims=True) / fp8_max, 1e-12)
    x_q = (xf / x_scale).astype(weight.dtype)
    acc = jax.lax.dot_general(
        x_q.astype(jnp.bfloat16),
        weight.astype(jnp.bfloat16),
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    n = weight.shape[1]
    ws = jnp.broadcast_to(weight_scale.astype(jnp.float32).reshape(-1), (n,))
    return (acc * x_scale * ws).astype(x.dtype)


def patch_small_m(module):
    module.fp8_linear_small_m = _fp8_linear_small_m


def patch_block_fp8(module):
    def blockwise_quantized_matmul_kernel(x, w_q, w_scale, *, block_size, x_q_dtype):
        """x [M, K], w_q [N, K], w_scale [K / block_size, 1, N]."""
        import jax.numpy as jnp

        m, k = x.shape
        n = w_q.shape[0]
        blocks = k // block_size
        xf = x.astype(jnp.float32).reshape(m, blocks, block_size)
        x_q, x_scale = module._quantize_block(xf, 2, x_q_dtype)  # [M, B, bs], [M, B, 1]
        w = w_q.reshape(n, blocks, block_size)
        dots = jnp.einsum(
            "mbk,nbk->bmn",
            x_q.astype(jnp.bfloat16),
            w.astype(jnp.bfloat16),
            preferred_element_type=jnp.float32,
        )  # [B, M, N]
        scaled = dots * x_scale.transpose(1, 0, 2) * w_scale.astype(jnp.float32)
        return scaled.sum(axis=0).astype(x.dtype)

    module._blockwise_quantized_matmul_kernel = blockwise_quantized_matmul_kernel


def _dequantize(weight, scale, dtype):
    """weight [G, K, N]; scale [G, K blocks, 1, N blocks], or None."""
    import jax.numpy as jnp

    if scale is None:
        return weight.astype(dtype)
    g, k, n = weight.shape
    kb, nb = scale.shape[1], scale.shape[3]
    blocks = weight.astype(jnp.float32).reshape(g, kb, k // kb, nb, n // nb)
    s = scale.astype(jnp.float32)[:, :, 0, :][:, :, None, :, None]
    return (blocks * s).reshape(g, k, n).astype(dtype)


def patch_fused_moe(module):
    def fused_moe_jax(
        hidden_states,  # [m, k]
        w1,  # [g, k, 2 * n]
        w2,  # [g, n, k]
        topk_weights,  # [m, topk]
        topk_ids,  # i32[m, topk], global expert ids; -1 marks a dropped slot
        w1_scale,  # optional [g, k blocks, 1, 2 * n blocks]
        w2_scale,  # optional [g, n blocks, 1, k blocks]
        w1_bias,  # optional [g, 1, 2 * n]
        w2_bias,  # optional [g, 1, k]
        topk,
        activation_name,
        no_combine=False,
        global_num_experts=0,
        expert_group_offset=0,
        use_sc_gather=0,
        w_quant_bits=0,
        permute_dense_threshold=0,
        w1_channel_scale=None,
        w2_channel_scale=None,
        expert_group_offset_tensor=None,
    ):
        import jax.numpy as jnp

        from sglang_tpu.moe.pallas.gmm_v2 import apply_act_fn

        del topk, use_sc_gather, permute_dense_threshold
        packed = w1.dtype in (jnp.uint8, jnp.int32) or w2.dtype in (jnp.uint8, jnp.int32)
        if w_quant_bits or packed or w1_channel_scale is not None or w2_channel_scale is not None:
            raise NotImplementedError(
                "fused MoE with packed FP4/int4 weights or channel scales needs the Pallas kernel"
            )

        num_experts = w1.shape[0]
        intermediate = w2.shape[1]
        out_dtype = hidden_states.dtype
        if expert_group_offset_tensor is not None:
            offset = expert_group_offset_tensor.astype(jnp.int32).reshape(())
        else:
            offset = expert_group_offset if global_num_experts > num_experts else 0

        # [m, topk, g]: which local expert each slot routes to, if any.
        local_ids = topk_ids.astype(jnp.int32) - offset
        experts = jnp.arange(num_experts, dtype=jnp.int32)
        routes = (local_ids[..., None] == experts).astype(jnp.float32)

        w1d = _dequantize(w1, w1_scale, out_dtype)
        w2d = _dequantize(w2, w2_scale, out_dtype)
        gate_up = jnp.einsum(
            "mk,gkn->gmn", hidden_states, w1d, preferred_element_type=jnp.float32
        )
        if w1_bias is not None:
            gate_up = gate_up + w1_bias.astype(jnp.float32)
        activated = apply_act_fn(
            gate_up, intermediate, module._fuse_act_name(activation_name)
        ).astype(out_dtype)
        expert_out = jnp.einsum(
            "gmn,gnk->gmk", activated, w2d, preferred_element_type=jnp.float32
        )
        if w2_bias is not None:
            expert_out = expert_out + w2_bias.astype(jnp.float32)

        if no_combine:
            # Each slot's unweighted output; zero for dropped and non-local slots.
            return jnp.einsum("mtg,gmk->mtk", routes, expert_out).astype(out_dtype)
        weights = jnp.einsum("mt,mtg->mg", topk_weights.astype(jnp.float32), routes)
        return jnp.einsum("mg,gmk->mk", weights, expert_out).astype(out_dtype)

    module._fused_moe_jax = fused_moe_jax


PATCHES = {
    SMALL_M_MODULE: patch_small_m,
    BLOCK_FP8_MODULE: patch_block_fp8,
    FUSED_MOE_MODULE: patch_fused_moe,
}
