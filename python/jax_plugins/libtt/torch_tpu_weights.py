"""Marks the module parameters of torch_tpu programs as weights.

libtt prepares a weight's device layout once and keeps it between calls, and a
traced program reads the weight in place, only when the program marks the
argument as a weight, as SGLang-JAX does with `tt.weight_dtype_override` and
tt-xla's PyTorch backend with `tt.mark_argument`. torch_tpu passes module
parameters as plain arguments, so in TPU compatibility mode a pass over each
inference graph marks the parameters it reads, before AOT Autograd.

`tpu_compat` imports this module before torch, so it imports torch lazily.
"""

_mark_weight = None


def _mark_weight_op():
    """`libtt::mark_weight(weight) -> weight`, a custom call that tt-xla folds away."""
    global _mark_weight
    if _mark_weight is not None:
        return _mark_weight

    import jax
    import torch
    from torch_tpu._internal import pallas

    def mark(weight):
        shape = weight.shape
        # As in SGLang-JAX and tt-xla's PyTorch backend, the custom call takes
        # a tensor of rank 3 or more.
        if weight.ndim < 3:
            weight = weight.reshape((1,) * (3 - weight.ndim) + shape)
        # Without a dtype of its own, the weight takes the one of the
        # experimental_weight_dtype compile option, if any.
        weight = jax.ffi.ffi_call(
            "tt.weight_dtype_override", jax.ShapeDtypeStruct(weight.shape, weight.dtype)
        )(weight)
        return weight.reshape(shape)

    kernel = pallas.custom_jax_kernel(mark, name="libtt_mark_weight", warn_deprecated=False)

    @torch.library.custom_op("libtt::mark_weight", mutates_args=(), device_types=["tpu"])
    def mark_weight(weight: torch.Tensor) -> torch.Tensor:
        return kernel(weight)

    @mark_weight.register_fake
    def _(weight):
        return torch.empty_like(weight)

    _mark_weight = torch.ops.libtt.mark_weight.default
    return _mark_weight


def _is_in_place(node):
    name = getattr(node.target, "__name__", node.target)
    return isinstance(name, str) and name.endswith("_") and not name.endswith("__")


def mark_weights(graph_module):
    """Marks the parameters an inference graph reads but doesn't update."""
    import torch

    if torch.is_grad_enabled():
        return
    graph = graph_module.graph
    placeholders = list(graph.find_nodes(op="placeholder"))
    weights = [
        node
        for node in placeholders
        if isinstance(node.meta.get("example_value"), torch.nn.Parameter)
        and node.users
        and not any(_is_in_place(user) for user in node.users)
    ]
    if not weights:
        return
    mark_weight = _mark_weight_op()
    with graph.inserting_after(placeholders[-1]):
        for weight in reversed(weights):
            users = list(weight.users)
            marked = graph.call_function(mark_weight, (weight,))
            marked.meta["example_value"] = weight.meta["example_value"]
            for user in users:
                user.replace_input_with(weight, marked)
    graph_module.recompile()


def patch_compiler(module):
    """Runs `mark_weights` after torch_tpu's own pre-grad passes."""
    execute_pre_grad_passes = module.Compiler.execute_pre_grad_passes

    def execute_pre_grad_passes_and_mark_weights(self, graph_module):
        execute_pre_grad_passes(self, graph_module)
        mark_weights(graph_module)

    module.Compiler.execute_pre_grad_passes = execute_pre_grad_passes_and_mark_weights


PATCHES = {"torch_tpu._internal.compile.compiler": patch_compiler}
