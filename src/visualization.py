"""
=========================================================
Privacy-Preserving Federated Learning Framework

Module : Visualization
Version: 3.0 (Final Dissertation Release)

Author : Khawar Akram
=========================================================
"""

# ==========================================================
# VISUALIZATION UTILITIES
# ==========================================================

import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import ConfusionMatrixDisplay

from config import FIGURES_PATH


# ==========================================================
# TRAINING CURVES
# ==========================================================

def plot_training_history(history):

    plt.figure(figsize=(8,5))

    plt.plot(
        history["train_accuracy"],
        label="Train Accuracy"
    )

    plt.plot(
        history["test_accuracy"],
        label="Test Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy (%)")

    plt.title("Training Accuracy")

    plt.legend()

    plt.grid(True)

    return plt.gcf()


# ==========================================================
# LOSS CURVES
# ==========================================================

def plot_loss_history(history):

    plt.figure(figsize=(8,5))

    plt.plot(
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        history["test_loss"],
        label="Test Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training Loss")

    plt.legend()

    plt.grid(True)

    return plt.gcf()


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

def plot_confusion_matrix(cm, class_names):

    fig, ax = plt.subplots(figsize=(8,8))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    ).plot(
        cmap="Blues",
        ax=ax,
        xticks_rotation=45
    )

    plt.tight_layout()

    return fig


# ==========================================================
# SAVE FIGURE
# ==========================================================

def save_figure(fig, filename):

    filepath = os.path.join(
        FIGURES_PATH,
        filename
    )

    fig.savefig(
        filepath,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"✅ Saved: {filename}")


# ==========================================================
# SAVE ALL VISUALIZATIONS
# ==========================================================

from sklearn.metrics import confusion_matrix

def save_visualizations(
    history,
    predictions,
    targets,
    class_names,
    experiment,
):
    """
    Save all experiment figures.

    Parameters
    ----------
    history : dict
    predictions : list
    targets : list
    class_names : list
    experiment : dict
    """

    experiment_id = experiment["id"]

    # ---------------------------------------------
    # Accuracy Curve
    # ---------------------------------------------

    acc_fig = plot_training_history(history)

    save_figure(
        acc_fig,
        f"{experiment_id}_Accuracy.png"
    )

    # ---------------------------------------------
    # Loss Curve
    # ---------------------------------------------

    loss_fig = plot_loss_history(history)

    save_figure(
        loss_fig,
        f"{experiment_id}_Loss.png"
    )

    # ---------------------------------------------
    # Confusion Matrix
    # ---------------------------------------------

    cm = confusion_matrix(
        targets,
        predictions
    )

    cm_fig = plot_confusion_matrix(
        cm,
        class_names
    )

    save_figure(
        cm_fig,
        f"{experiment_id}_ConfusionMatrix.png"
    )

    plt.close("all")

    print("✅ Visualizations Saved")