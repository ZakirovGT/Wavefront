"""Public API with lazy imports to keep training and inference independent."""

from importlib import import_module

_EXPORTS = {
    "load_three_stage_run_kwargs": (
        "wavefront.pipelines.config",
        "load_three_stage_run_kwargs",
    ),
    "ThreeStageDataConfig": ("wavefront.pipelines.three_stage", "ThreeStageDataConfig"),
    "run_three_stage_pipeline": (
        "wavefront.pipelines.three_stage",
        "run_three_stage_pipeline",
    ),
    "TwoStageDataConfig": ("wavefront.pipelines.two_stage", "TwoStageDataConfig"),
    "run_two_stage_pipeline": (
        "wavefront.pipelines.two_stage",
        "run_two_stage_pipeline",
    ),
    "load_two_stage_run_kwargs": (
        "wavefront.pipelines.two_stage_config",
        "load_two_stage_run_kwargs",
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
