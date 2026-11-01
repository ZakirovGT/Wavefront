"""Public API with lazy imports to keep training and inference independent."""

from importlib import import_module

_EXPORTS = {
    "ModernDeepONet": ("wavefront.models.deeponet", "ModernDeepONet"),
    "FNO2d": ("wavefront.models.fno", "FNO2d"),
    "setup_deeponet": ("wavefront.models.factory", "setup_deeponet"),
    "setup_fno": ("wavefront.models.factory", "setup_fno"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value
