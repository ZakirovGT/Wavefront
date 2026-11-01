"""Public API with lazy imports to keep training and inference independent."""

from importlib import import_module

_EXPORTS = {
    "generate_distortion_span_dataset": (
        "wavefront.data.generators",
        "generate_distortion_span_dataset",
    ),
    "generate_mixed_span_dataset": (
        "wavefront.data.generators",
        "generate_mixed_span_dataset",
    ),
    "generate_spiral_span_dataset": (
        "wavefront.data.generators",
        "generate_spiral_span_dataset",
    ),
    "generate_zernike_span_dataset": (
        "wavefront.data.generators",
        "generate_zernike_span_dataset",
    ),
    "interpolate_sensor_derivatives_to_grid": (
        "wavefront.data.interpolation",
        "interpolate_sensor_derivatives_to_grid",
    ),
    "load_sensor_coords_from_csv": (
        "wavefront.data.sensors",
        "load_sensor_coords_from_csv",
    ),
    "DataGenerator": ("wavefront.data.batching", "DataGenerator"),
    "GridDataGenerator": ("wavefront.data.batching", "GridDataGenerator"),
    "generate_one_res_training_data": (
        "wavefront.data.deeponet_data",
        "generate_one_res_training_data",
    ),
    "make_supervised_task": ("wavefront.data.deeponet_data", "make_supervised_task"),
    "make_test_task": ("wavefront.data.deeponet_data", "make_test_task"),
    "prepare_fno_arrays": ("wavefront.data.fno_inputs", "prepare_fno_arrays"),
    "load_branch_sensor_grid": ("wavefront.data.sensors", "load_branch_sensor_grid"),
    "SyntheticDataset": ("wavefront.data.synthetic", "SyntheticDataset"),
    "SyntheticDatasetConfig": ("wavefront.data.synthetic", "SyntheticDatasetConfig"),
    "generate_synthetic_dataset": (
        "wavefront.data.synthetic",
        "generate_synthetic_dataset",
    ),
    "load_synthetic_dataset_config": (
        "wavefront.data.synthetic",
        "load_synthetic_dataset_config",
    ),
    "DATASET_SCHEMA_VERSION": (
        "wavefront.data.dataset_artifacts",
        "DATASET_SCHEMA_VERSION",
    ),
    "DatasetArtifact": ("wavefront.data.dataset_artifacts", "DatasetArtifact"),
    "generate_and_save_synthetic_dataset": (
        "wavefront.data.dataset_artifacts",
        "generate_and_save_synthetic_dataset",
    ),
    "load_synthetic_dataset_artifact": (
        "wavefront.data.dataset_artifacts",
        "load_synthetic_dataset_artifact",
    ),
    "save_synthetic_dataset_artifact": (
        "wavefront.data.dataset_artifacts",
        "save_synthetic_dataset_artifact",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value
