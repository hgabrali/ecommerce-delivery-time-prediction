"""
src/__init__.py
---------------
Package initialization for the ecommerce delivery time prediction project.

Exposes the main pipeline functions and model utilities for convenient imports.

Author: hgabrali
Date: 2026
"""

from .data_preprocessing import (
    load_data,
    engineer_features,
    clean_data,
    encode_and_scale,
    split_dataset,
    run_preprocessing_pipeline,
)
from .model_training import (
    train_all_models,
    evaluate_model,
    save_model,
    load_model,
    tune_xgboost,
)
from .visualization import (
    plot_target_distribution,
    plot_correlation_heatmap,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_model_comparison,
    plot_feature_importance,
)

__all__ = [
    "load_data",
    "engineer_features",
    "clean_data",
    "encode_and_scale",
    "split_dataset",
    "run_preprocessing_pipeline",
    "train_all_models",
    "evaluate_model",
    "save_model",
    "load_model",
    "tune_xgboost",
    "plot_target_distribution",
    "plot_correlation_heatmap",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_model_comparison",
    "plot_feature_importance",
]

__version__ = "1.0.0"
__author__ = "hgabrali"
