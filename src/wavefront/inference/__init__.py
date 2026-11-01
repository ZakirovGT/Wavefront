"""Public API with lazy imports to keep training and inference independent."""

from importlib import import_module

_EXPORTS = {
    "ModelArtifacts": ("wavefront.inference.artifacts", "ModelArtifacts"),
    "load_artifacts": ("wavefront.inference.artifacts", "load_artifacts"),
    "predict_batch_on_grid": ("wavefront.inference.deeponet", "predict_batch_on_grid"),
    "save_predictions": ("wavefront.inference.deeponet", "save_predictions"),
    "load_predictor": ("wavefront.inference.predictor", "load_predictor"),
    "predict_from_csv": ("wavefront.inference.predictor", "predict_from_csv"),
    "predict_from_dataframes": (
        "wavefront.inference.predictor",
        "predict_from_dataframes",
    ),
    "predict_fno_from_gradient_grids": (
        "wavefront.inference.operators",
        "predict_fno_from_gradient_grids",
    ),
    "predict_fno_from_sensor_gradients": (
        "wavefront.inference.operators",
        "predict_fno_from_sensor_gradients",
    ),
    "predict_metamodel_from_sensor_gradients": (
        "wavefront.inference.operators",
        "predict_metamodel_from_sensor_gradients",
    ),
    "save_inference_outputs": ("wavefront.inference.outputs", "save_inference_outputs"),
    "MeasurementColumns": ("wavefront.inference.measurements", "MeasurementColumns"),
    "SensorGradientBatch": ("wavefront.inference.measurements", "SensorGradientBatch"),
    "SensorLayout": ("wavefront.inference.measurements", "SensorLayout"),
    "denormalize_wavefronts": (
        "wavefront.inference.measurements",
        "denormalize_wavefronts",
    ),
    "load_sensor_gradient_csv": (
        "wavefront.inference.measurements",
        "load_sensor_gradient_csv",
    ),
    "load_sensor_layout": ("wavefront.inference.measurements", "load_sensor_layout"),
    "normalize_sensor_gradients": (
        "wavefront.inference.measurements",
        "normalize_sensor_gradients",
    ),
    "save_physical_wavefront_outputs": (
        "wavefront.inference.outputs",
        "save_physical_wavefront_outputs",
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
