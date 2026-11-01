"""Exercise public CLIs, checkpoints and portable artifacts on a small CPU run."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def run_script(name, *args):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), *map(str, args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.mark.parametrize(
    "script", sorted(p.name for p in (ROOT / "scripts").glob("*.py"))
)
def test_cli_help(script):
    assert "usage:" in run_script(script, "--help")


@pytest.fixture(scope="module")
def workflow(tmp_path_factory):
    root = tmp_path_factory.mktemp("workflow")
    dataset = root / "dataset"
    run_script(
        "generate_dataset.py",
        "--config",
        "configs/smoke/standalone_data.yaml",
        "--output-dir",
        dataset,
    )
    runs = {}
    for operator in ["fno", "deeponet"]:
        run_script(
            f"train_{operator}.py",
            "--config-dir",
            "configs/smoke",
            "--dataset-dir",
            dataset,
            "--results-root",
            root / "runs",
        )
        runs[operator] = next(
            (root / "runs" / operator).glob("*/run_manifest.json")
        ).parent
    for operator in ["two_stage", "three_stage"]:
        run_script(
            f"train_{operator}.py",
            "--config",
            f"configs/smoke/{operator}.yaml",
            "--dataset-dir",
            dataset,
            "--results-root",
            root / operator,
        )
        runs[operator] = next((root / operator).glob("*/manifest.json")).parent
    # Move runs and the dataset, then discard the old paths. Inference must
    # use saved coordinates, configuration and parameters on this machine.
    moved = root / "moved"
    moved.mkdir()
    for name, path in runs.items():
        runs[name] = Path(shutil.move(str(path), moved / name))
    dataset = Path(shutil.move(str(dataset), moved / "dataset"))
    return root, dataset, runs


def test_dataset_portability_and_checksum(workflow):
    from wavefront.data.dataset_artifacts import load_synthetic_dataset_artifact

    root, dataset, _ = workflow
    artifact = load_synthetic_dataset_artifact(dataset)
    assert artifact.wavefronts.shape == (9, 64)
    assert artifact.sensor_gradients.shape == (9, 483, 2)
    assert artifact.config.branch_grid_path == str(dataset / "sensor_layout.csv")
    assert np.isfinite(artifact.grid_gradients).all()
    corrupt = root / "corrupt"
    shutil.copytree(dataset, corrupt)
    with (corrupt / "arrays.npz").open("ab") as f:
        f.write(b"corrupted")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_synthetic_dataset_artifact(corrupt)


@pytest.mark.parametrize("operator", ["deeponet", "fno", "two_stage", "three_stage"])
def test_csv_inference_after_moving_run(workflow, operator):
    root, _, runs = workflow
    output = root / "predictions" / operator
    run_script(
        f"predict_{operator}.py",
        "--run-dir",
        runs[operator],
        "--sensor-layout",
        "1.csv",
        "--measurements",
        "1.csv",
        "--gx-column",
        "Zx",
        "--gy-column",
        "Zy",
        "--sensor-id-column",
        "N",
        "--layout-sensor-id-column",
        "N",
        "--normalization-scale",
        "1",
        "--output-dir",
        output,
    )
    manifest = json.loads((output / "inference_manifest.json").read_text())
    assert manifest["n_sensors"] == 483
    predictions = np.load(output / "wavefront_predictions_normalized.npy")
    assert predictions.shape == (1, 8, 8)
    assert np.isfinite(predictions).all()
    assert list(output.rglob("*.png"))


def test_all_models_benchmark(workflow):
    root, dataset, runs = workflow
    config = yaml.safe_load((ROOT / "configs/smoke/benchmark.yaml").read_text())
    config["runs"] = {name: str(path) for name, path in runs.items()}
    path = root / "benchmark.yaml"
    path.write_text(yaml.safe_dump(config))
    run_script(
        "run_benchmark.py",
        "--config",
        path,
        "--dataset-dir",
        dataset,
        "--output-root",
        root / "benchmark",
        "--run-name",
        "test",
        "--fail-fast",
    )
    report = json.loads((root / "benchmark/test/report.json").read_text())
    assert not report["failures"]
    assert set(report["results"]) == {"poisson", *runs}
    assert list((root / "benchmark/test").rglob("*.png"))


@pytest.mark.parametrize("operator", ["two_stage", "three_stage"])
def test_pipeline_generates_its_own_data(tmp_path, operator):
    # This branch uses separate seeds and must work without --dataset-dir too.
    run_script(
        f"train_{operator}.py",
        "--config",
        f"configs/smoke/{operator}.yaml",
        "--results-root",
        tmp_path,
    )
    manifest = json.loads(next(tmp_path.glob("*/manifest.json")).read_text())
    assert manifest["dataset"]["source"] == "generated_in_process"
