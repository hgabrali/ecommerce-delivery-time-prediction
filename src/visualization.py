"""
visualization.py
----------------
Plotting and visualization utilities for the e-commerce delivery time
prediction project.

Author: hgabrali
Date: 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from typing import Any, Optional
import os

OUTPUT_DIR = "reports/figures"


def _ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save_fig(fig, filename: str) -> None:
    _ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_target_distribution(y, save: bool = True):
    """Bar and pie chart of the binary delivery outcome."""
    counts = y.value_counts()
    labels = ["On-Time (0)", "Delayed (1)"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(labels, counts.values, color=["#2196F3", "#F44336"], edgecolor="black")
    axes[0].set_title("Delivery Outcome - Count", fontsize=14)
    axes[0].set_ylabel("Number of Orders")
    axes[1].pie(counts.values, labels=labels, autopct="%1.1f%%",
                colors=["#2196F3", "#F44336"], startangle=90)
    axes[1].set_title("Delivery Outcome - Proportion", fontsize=14)
    fig.suptitle("Target Variable Distribution", fontsize=16, fontweight="bold")
    plt.tight_layout()
    if save:
        _save_fig(fig, "target_distribution.png")
    return fig


def plot_correlation_heatmap(df, save: bool = True):
    """Correlation heatmap for all numeric features."""
    corr = df.select_dtypes(include=[np.number]).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
    plt.tight_layout()
    if save:
        _save_fig(fig, "correlation_heatmap.png")
    return fig


def plot_confusion_matrix(model, X_test, y_test,
                           model_name: str = "Model", save: bool = True):
    """Normalized confusion matrix display."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, normalize="true", cmap="Blues",
        display_labels=["On-Time", "Delayed"], ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        _save_fig(fig, f"cm_{model_name.lower().replace(' ', '_')}.png")
    return fig


def plot_roc_curve(model, X_test, y_test,
                   model_name: str = "Model", save: bool = True):
    """ROC curve with AUC annotation."""
    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_estimator(model, X_test, y_test, name=model_name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    ax.set_title(f"ROC Curve - {model_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save:
        _save_fig(fig, f"roc_{model_name.lower().replace(' ', '_')}.png")
    return fig


def plot_model_comparison(results: dict, metrics=None, save: bool = True):
    """Grouped bar chart comparing models across multiple metrics."""
    if metrics is None:
        metrics = ["accuracy", "f1_weighted", "roc_auc"]
    models = list(results.keys())
    x = np.arange(len(models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#42A5F5", "#66BB6A", "#FFA726"]
    for i, metric in enumerate(metrics):
        values = [results[m].get(metric) or 0 for m in models]
        bars = ax.bar(x + i * width, values, width,
                      label=metric.replace("_", " ").title(), color=colors[i])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    plt.tight_layout()
    if save:
        _save_fig(fig, "model_comparison.png")
    return fig


def plot_feature_importance(model, feature_names: list,
                             model_name: str = "Model",
                             top_n: int = 20, save: bool = True):
    """Horizontal bar chart of top-N feature importances."""
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(f"{model_name} does not expose feature_importances_.")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n), importances[indices][::-1],
            color="#78909C", edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]], fontsize=10)
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_n} Feature Importances - {model_name}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        _save_fig(fig, f"fi_{model_name.lower().replace(' ', '_')}.png")
    return fig


def interactive_delivery_timeline(df):
    """Interactive Plotly line chart of daily orders by delivery outcome."""
    df = df.copy()
    df["purchase_date"] = pd.to_datetime(df["order_purchase_timestamp"]).dt.date
    daily = (df.groupby(["purchase_date", "delivery_time_class"])
             .size().reset_index(name="count"))
    daily["outcome"] = daily["delivery_time_class"].map({0: "On-Time", 1: "Delayed"})
    fig = px.line(daily, x="purchase_date", y="count", color="outcome",
                  title="Daily Orders by Delivery Outcome",
                  color_discrete_map={"On-Time": "#2196F3", "Delayed": "#F44336"})
    fig.update_layout(template="plotly_white", hovermode="x unified")
    return fig

