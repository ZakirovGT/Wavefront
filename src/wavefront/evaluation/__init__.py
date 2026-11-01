"""Public API with lazy imports to keep training and inference independent."""

from importlib import import_module

_EXPORTS = {
    "EvalConfig": ("wavefront.evaluation.config", "EvalConfig"),
    "get_grid_inputs": ("wavefront.evaluation.data", "get_grid_inputs"),
    "get_sensor_coords": ("wavefront.evaluation.data", "get_sensor_coords"),
    "take_common_test": ("wavefront.evaluation.data", "take_common_test"),
    "get_error": ("wavefront.evaluation.deeponet", "get_error"),
    "evaluate_deeponet_state": (
        "wavefront.evaluation.operators",
        "evaluate_deeponet_state",
    ),
    "evaluate_fno_state": ("wavefront.evaluation.operators", "evaluate_fno_state"),
    "evaluate_metamodel_state": (
        "wavefront.evaluation.operators",
        "evaluate_metamodel_state",
    ),
    "evaluate_poisson_baseline": (
        "wavefront.evaluation.operators",
        "evaluate_poisson_baseline",
    ),
    "add_result": ("wavefront.evaluation.reporting", "add_result"),
    "create_report": ("wavefront.evaluation.reporting", "create_report"),
    "print_report": ("wavefront.evaluation.reporting", "print_report"),
    "benchmark_inference_deeponet": (
        "wavefront.evaluation.timing",
        "benchmark_inference_deeponet",
    ),
    "benchmark_inference_fno": (
        "wavefront.evaluation.timing",
        "benchmark_inference_fno",
    ),
    "benchmark_inference_poisson": (
        "wavefront.evaluation.timing",
        "benchmark_inference_poisson",
    ),
    "load_standalone_deeponet_state": (
        "wavefront.evaluation.loaders",
        "load_standalone_deeponet_state",
    ),
    "load_standalone_fno_state": (
        "wavefront.evaluation.loaders",
        "load_standalone_fno_state",
    ),
    "load_three_stage_metamodel_state": (
        "wavefront.evaluation.loaders",
        "load_three_stage_metamodel_state",
    ),
    "load_two_stage_metamodel_state": (
        "wavefront.evaluation.loaders",
        "load_two_stage_metamodel_state",
    ),
    "release_jax_memory": ("wavefront.evaluation.loaders", "release_jax_memory"),
    "run_sequential_benchmark": (
        "wavefront.evaluation.runner",
        "run_sequential_benchmark",
    ),
    "BenchmarkDataConfig": (
        "wavefront.evaluation.benchmark_config",
        "BenchmarkDataConfig",
    ),
    "BenchmarkRunConfig": (
        "wavefront.evaluation.benchmark_config",
        "BenchmarkRunConfig",
    ),
    "load_benchmark_config": (
        "wavefront.evaluation.benchmark_config",
        "load_benchmark_config",
    ),
    "relative_error_map": (
        "wavefront.evaluation.benchmark_visualization",
        "relative_error_map",
    ),
    "save_benchmark_comparison": (
        "wavefront.evaluation.benchmark_visualization",
        "save_benchmark_comparison",
    ),
    "save_stage1_gradient_comparison": (
        "wavefront.evaluation.benchmark_visualization",
        "save_stage1_gradient_comparison",
    ),
    "as_gradient_grid": ("wavefront.evaluation.sample_predictions", "as_gradient_grid"),
    "predict_deeponet_sample": (
        "wavefront.evaluation.sample_predictions",
        "predict_deeponet_sample",
    ),
    "predict_fno_sample": (
        "wavefront.evaluation.sample_predictions",
        "predict_fno_sample",
    ),
    "predict_metamodel_sample": (
        "wavefront.evaluation.sample_predictions",
        "predict_metamodel_sample",
    ),
    "predict_poisson_sample": (
        "wavefront.evaluation.sample_predictions",
        "predict_poisson_sample",
    ),
    "select_deeponet_input": (
        "wavefront.evaluation.sample_predictions",
        "select_deeponet_input",
    ),
    "BenchmarkVisualizationConfig": (
        "wavefront.evaluation.benchmark_config",
        "BenchmarkVisualizationConfig",
    ),
    "predict_metamodel_gradient_sample": (
        "wavefront.evaluation.sample_predictions",
        "predict_metamodel_gradient_sample",
    ),
    "run_sequential_sample_visualization": (
        "wavefront.evaluation.sample_runner",
        "run_sequential_sample_visualization",
    ),
    "visualize": ("wavefront.evaluation.visualization", "visualize"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value
