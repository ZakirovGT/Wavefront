"""Public API with lazy imports to keep training and inference independent."""

from importlib import import_module

_EXPORTS = {
    "DeepONetGradMapConfig": ("wavefront.metamodel.gradmap", "DeepONetGradMapConfig"),
    "deeponet_predict_grad_grid": (
        "wavefront.metamodel.gradmap",
        "deeponet_predict_grad_grid",
    ),
    "eval_deeponet_gradmap_full_grid_error": (
        "wavefront.metamodel.gradmap",
        "eval_deeponet_gradmap_full_grid_error",
    ),
    "setup_deeponet_gradmap": ("wavefront.metamodel.gradmap", "setup_deeponet_gradmap"),
    "train_deeponet_gradmap": ("wavefront.metamodel.gradmap", "train_deeponet_gradmap"),
    "train_fno_on_deeponet_outputs": (
        "wavefront.metamodel.stage2_fno",
        "train_fno_on_deeponet_outputs",
    ),
    "eval_joint_deeponet_fno_error": (
        "wavefront.metamodel.joint",
        "eval_joint_deeponet_fno_error",
    ),
    "finetune_joint_deeponet_fno": (
        "wavefront.metamodel.joint",
        "finetune_joint_deeponet_fno",
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
