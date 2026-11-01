import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from wavefront.config import load_config
from wavefront.models.factory import setup_deeponet, setup_fno
from wavefront.training.deeponet import apply_net
from wavefront.training.fno import apply_fno

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "package",
    [
        "physics",
        "data",
        "training",
        "models",
        "metamodel",
        "evaluation",
        "inference",
        "pipelines",
    ],
)
def test_public_exports(package):
    module = importlib.import_module(f"wavefront.{package}")
    for name in module.__all__:
        assert getattr(module, name) is not None


@pytest.mark.parametrize("operator", ["deeponet", "fno"])
def test_dropout_training_and_deterministic_inference(operator):
    args = load_config(operator, ROOT / "configs/smoke", dropout_rate=0.2, n_sensors=3)
    key = jax.random.PRNGKey(42)
    if operator == "fno":
        _, model, _, params = setup_fno(args, key)
        inputs = [jnp.ones((2, 8, 8, 2))]
        apply = apply_fno
    else:
        _, model, _, params = setup_deeponet(args, key)
        inputs = [jnp.ones((2, 6)), jnp.ones(2), jnp.ones(2)]
        apply = apply_net
    training = apply(model, params, *inputs, rng=key, training=True)
    first = apply(model, params, *inputs, training=False)
    second = apply(model, params, *inputs, training=False)
    assert np.isfinite(training).all()
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(training, first)


def test_layout_coordinates_match_training():
    from wavefront.data.sensors import load_branch_sensor_grid
    from wavefront.inference.measurements import load_sensor_layout

    pixel_layout = load_sensor_layout(ROOT / "1.csv", coordinate_system="pixel")
    np.testing.assert_allclose(
        pixel_layout.coordinates, load_branch_sensor_grid(ROOT / "1.csv")
    )


def test_unbatched_fno_grid():
    from wavefront.inference.operators import predict_fno_from_gradient_grids

    args = load_config("fno", ROOT / "configs/smoke")
    _, model, _, params = setup_fno(args, jax.random.PRNGKey(1))
    state = {"args": args, "model_fn": model, "params": params}
    field = np.ones((8, 8, 2), dtype=np.float32)
    single = predict_fno_from_gradient_grids(state, field)
    batched = predict_fno_from_gradient_grids(state, field[None])
    np.testing.assert_allclose(single, batched)


def test_disabled_checkpoints(tmp_path):
    from wavefront.data.dataset_artifacts import generate_and_save_synthetic_dataset
    from wavefront.data.synthetic import load_synthetic_dataset_config
    from wavefront.training.standalone_runner import run_standalone_training

    config = load_synthetic_dataset_config(ROOT / "configs/smoke/standalone_data.yaml")
    config.branch_grid_path = str(ROOT / "1.csv")
    artifact = generate_and_save_synthetic_dataset(tmp_path / "dataset", config=config)
    result = run_standalone_training(
        "deeponet",
        config_dir=ROOT / "configs/smoke",
        result_parent=tmp_path / "runs",
        dataset_artifact=artifact,
        model_overrides={"steps": 1, "checkpoint_iter": 0},
    )
    assert (result / "artifacts/params").is_dir()
    assert not (result / "ckpt").exists()
