"""Exported programs carry the TTIR -> TTNN pipeline they were compiled with.

With export_path, the plugin writes each program's input to the TTIR -> TTNN
pipeline as irs/ttir_to_ttnn_*.mlir. Its first line is an MLIR comment holding
that pipeline with every option the plugin set, so ttmlir-opt can rerun the
compile offline (README.md, "Compile a program offline").
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("level", ["O0", "O1"])
def test_exported_pipeline(tmp_path, trace, level):
    device = jax.devices("tt")[0]
    run = jax.jit(
        lambda x: jnp.exp(x * 0.25) + 1.0,
        compiler_options={
            "optimization_level": level,
            "enable_trace": str(trace).lower(),
            "export_path": str(tmp_path),
        },
    )
    values = np.linspace(-2.0, 2.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    actual = run(jax.device_put(values, device))
    np.testing.assert_allclose(np.asarray(actual), np.exp(values * 0.25) + 1.0, rtol=2e-2)

    files = sorted((tmp_path / "irs").glob("ttir_to_ttnn_*.mlir"))
    assert len(files) == 1, files
    header, module = files[0].read_text().split("\n", 1)
    prefix = "// --ttir-to-ttnn-common-pipeline="
    assert header.startswith(prefix), header
    # The options that differ from tt-mlir's defaults.
    options = dict(option.split("=", 1) for option in header[len(prefix) :].split())
    assert options.get("enable-trace", "false") == str(trace).lower()
    assert options.get("optimization-level", "0") == level[1:]
    # What the plugin always sets, unlike tt-mlir's defaults.
    assert options["enable-erase-inverse-ops-pass"] == "false"
    assert options["enable-eltwise-activation-fusion"] == "true"
    assert "func.func" in module
