"""Public API with lazy imports to keep training and inference independent."""

from importlib import import_module

_EXPORTS = {
    "apply_net": ("wavefront.training.deeponet", "apply_net"),
    "apply_net_task": ("wavefront.training.deeponet", "apply_net_task"),
    "apply_net_tasks": ("wavefront.training.deeponet", "apply_net_tasks"),
    "rel_l2_batch_loss": ("wavefront.training.deeponet", "rel_l2_batch_loss"),
    "step": ("wavefront.training.deeponet", "step"),
    "apply_fno": ("wavefront.training.fno", "apply_fno"),
    "rel_l2_field_loss": ("wavefront.training.fno", "rel_l2_field_loss"),
    "step_fno": ("wavefront.training.fno", "step_fno"),
    "loss_fn": ("wavefront.training.deeponet_losses", "loss_fn"),
    "loss_ics": ("wavefront.training.deeponet_losses", "loss_ics"),
    "loss_res": ("wavefront.training.deeponet_losses", "loss_res"),
    "fno_rel_l2": ("wavefront.training.fno_losses", "fno_rel_l2"),
    "loss_fno": ("wavefront.training.fno_losses", "loss_fno"),
    "loss_res_fno": ("wavefront.training.fno_losses", "loss_res_fno"),
    "set_mixed_precision": ("wavefront.training.precision", "set_mixed_precision"),
    "main_routine_deeponet": (
        "wavefront.training.deeponet_trainer",
        "main_routine_deeponet",
    ),
    "main_routine_fno": ("wavefront.training.fno_trainer", "main_routine_fno"),
    "main_routine": ("wavefront.training.runner", "main_routine"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value
