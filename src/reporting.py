"""
=========================================================
Privacy-Preserving Federated Learning Framework

Module : Reporting
Version: 3.0 (Final Dissertation Release)

Author : Khawar Akram
=========================================================
"""

from logging import log
import os
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)

# -------------------------------------------------------
# Result Directories
# -------------------------------------------------------

from pathlib import Path

PROJECT_ROOT = Path("/content/drive/MyDrive/COM748_Project")
BASE_RESULTS_DIR = PROJECT_ROOT / "06_Results"

TABLE_DIR = BASE_RESULTS_DIR / "Tables"
FIGURE_DIR = BASE_RESULTS_DIR / "Figures"
REPORT_DIR = BASE_RESULTS_DIR / "Reports"
CHECKPOINT_DIR = BASE_RESULTS_DIR / "Checkpoints"

# Optional (recommended)
HISTORY_DIR = TABLE_DIR / "Round_History"

# Existing master experiment log
# EXPERIMENT_LOG = TABLE_DIR / "experiment_log.csv"

# -------------------------------------------------------
# Federated Experiment Log
# -------------------------------------------------------

FEDERATED_EXPERIMENT_LOG = (
    TABLE_DIR / "federated_experiment_log.csv"
)

# -------------------------------------------------------
# Create directory structure
# -------------------------------------------------------

def create_result_directories():
    """
    Creates all required result directories.
    Safe to call every experiment.
    """

    directories = [

        BASE_RESULTS_DIR,

        TABLE_DIR,

        FIGURE_DIR,

        REPORT_DIR,

        CHECKPOINT_DIR,

        HISTORY_DIR

    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    print("Result directories verified.")

    # -------------------------------------------------------
# Timer utilities
# -------------------------------------------------------

def start_timer():

    return time.time()


def stop_timer(start_time):

    elapsed = time.time() - start_time

    return round(elapsed / 60, 2)

# -------------------------------------------------------
# Experiment Naming
# -------------------------------------------------------

def experiment_name(
        experiment_id,
        dp_enabled=False,
        noise=0.0,
        clip=0.0
):
    """
    Creates a readable experiment name.
    """

    if not dp_enabled:

        return experiment_id

    return (
        f"{experiment_id}"
        f"_Clip{clip}"
        f"_Noise{noise}"
    )

# -------------------------------------------------------
# Round History
# -------------------------------------------------------

def initialise_history():

    history = {

        "round": [],

        "train_loss": [],

        "test_loss": [],

        "accuracy": [],

        "best_accuracy": []

    }

    return history

# -------------------------------------------------------
# Append round statistics
# -------------------------------------------------------

def update_history(

    history,

    round_number,

    train_loss,

    test_loss,

    accuracy,

    best_accuracy

):

    history["round"].append(round_number)

    history["train_loss"].append(train_loss)

    history["test_loss"].append(test_loss)

    history["accuracy"].append(accuracy)

    history["best_accuracy"].append(best_accuracy)

# -------------------------------------------------------
# Convert history to DataFrame
# -------------------------------------------------------

def history_dataframe(history):

    return pd.DataFrame(history)


# -------------------------------------------------------
# Save Round History
# -------------------------------------------------------

def save_round_history(history, experiment_name):
    """
    Saves communication-round statistics to CSV.
    """

    create_result_directories()

    df = history_dataframe(history)

    filename = HISTORY_DIR / f"{experiment_name}_history.csv"

    df.to_csv(filename, index=False)

    print(f"✓ Round history saved: {filename}")

    return filename

# -------------------------------------------------------
# Load Experiment Log
# -------------------------------------------------------

def load_experiment_log():

    create_result_directories()

  #  if EXPERIMENT_LOG.exists():

   #     return pd.read_csv(EXPERIMENT_LOG)

    if FEDERATED_EXPERIMENT_LOG.exists():

        return pd.read_csv(FEDERATED_EXPERIMENT_LOG)

    columns = [

        "Experiment",

        "Method",

        "Clients",

        "Rounds",

        "LocalEpochs",

        "LearningRate",

        "BatchSize",

        "ClipNorm",

        "NoiseMultiplier",

        "BestAccuracy",

        "FinalAccuracy",

        "Precision",

        "Recall",

        "F1Score",

        "TrainingTime(min)"

    ]

    return pd.DataFrame(columns=columns)

# -------------------------------------------------------
# Append Experiment Log
# -------------------------------------------------------

def update_experiment_log(

        experiment,

        method,

        clients,

        rounds,

        local_epochs,

        learning_rate,

        batch_size,

        clip_norm,

        noise,

        best_accuracy,

        final_accuracy,

        precision,

        recall,

        f1,

        training_time

):

    log = load_experiment_log()

    new_row = {

        "Experiment": experiment,

        "Method": method,

        "Clients": clients,

        "Rounds": rounds,

        "LocalEpochs": local_epochs,

        "LearningRate": learning_rate,

        "BatchSize": batch_size,

        "ClipNorm": clip_norm,

        "NoiseMultiplier": noise,

        "BestAccuracy": round(best_accuracy,4),

        "FinalAccuracy": round(final_accuracy,4),

        "Precision": round(precision,4),

        "Recall": round(recall,4),

        "F1Score": round(f1,4),

        "TrainingTime(min)": training_time

    }

    log.loc[len(log)] = new_row

   # log.to_csv(EXPERIMENT_LOG, index=False)

    log.to_csv(
        FEDERATED_EXPERIMENT_LOG,
        index=False
    )

    # print("✓ experiment_log.csv updated")

    print(
        f"✓ Federated experiment log updated:\n"
        f"{FEDERATED_EXPERIMENT_LOG}"
    )


# -------------------------------------------------------
# Save Experiment Metadata
# -------------------------------------------------------

def save_metadata(summary, experiment):

    create_result_directories()

    filename = REPORT_DIR / f"{experiment}_metadata.json"

    with open(filename, "w") as f:

        json.dump(summary, f, indent=4)

    print("✓ Metadata saved")


# -------------------------------------------------------
# Save Text Summary
# -------------------------------------------------------

def save_experiment_summary(summary, experiment):

    create_result_directories()

    filename = REPORT_DIR / f"{experiment}_Summary.txt"

    with open(filename, "w") as f:

        f.write("="*60 + "\n")
        f.write("FEDERATED LEARNING EXPERIMENT SUMMARY\n")
        f.write("="*60 + "\n\n")

        for key, value in summary.items():

            f.write(f"{key:<30}: {value}\n")

    print(f"✓ Summary saved: {filename}")

# -------------------------------------------------------
# Build Summary Dictionary
# -------------------------------------------------------

def build_summary(

        experiment,

        method,

        clients,

        rounds,

        local_epochs,

        learning_rate,

        batch_size,

        clip_norm,

        noise,

        best_accuracy,

        final_accuracy,

        precision,

        recall,

        f1,

        training_time

):

    return {

        "Experiment": experiment,

        "Method": method,

        "Clients": clients,

        "Rounds": rounds,

        "Local Epochs": local_epochs,

        "Learning Rate": learning_rate,

        "Batch Size": batch_size,

        "Clip Norm": clip_norm,

        "Noise Multiplier": noise,

        "Best Accuracy": best_accuracy,

        "Final Accuracy": final_accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "Training Time (min)": training_time

    }

# -------------------------------------------------------
# Accuracy Curve
# -------------------------------------------------------

def plot_accuracy_curve(history, experiment):

    create_result_directories()

    plt.figure(figsize=(8,5))

    plt.plot(
        history["round"],
        history["accuracy"],
        color="royalblue",
        linewidth=2,
        marker="o",
        markersize=4,
        label="Test Accuracy"
    )

    plt.plot(
        history["round"],
        history["best_accuracy"],
        linestyle="--",
        linewidth=2,
        color="green",
        label="Best Accuracy"
    )

    plt.xlabel("Communication Round")

    plt.ylabel("Accuracy (%)")

    plt.title(f"{experiment} Accuracy")

    plt.grid(True, alpha=0.3)

    plt.legend()

    plt.tight_layout()

    filename = FIGURE_DIR / f"{experiment}_Accuracy.png"

    plt.savefig(filename, dpi=300)

    plt.close()

    print(f"✓ Accuracy figure saved")

# -------------------------------------------------------
# Loss Curve
# -------------------------------------------------------

def plot_loss_curve(history, experiment):

    create_result_directories()

    plt.figure(figsize=(8,5))

    plt.plot(

        history["round"],

        history["train_loss"],

        marker="o",

        linewidth=2,

        label="Training Loss"

    )

    plt.plot(

        history["round"],

        history["test_loss"],

        marker="s",

        linewidth=2,

        label="Test Loss"

    )

    plt.xlabel("Communication Round")

    plt.ylabel("Loss")

    plt.title(f"{experiment} Loss")

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()

    filename = FIGURE_DIR / f"{experiment}_Loss.png"

    plt.savefig(filename,dpi=300)

    plt.close()

    print("✓ Loss figure saved")

# -------------------------------------------------------
# Confusion Matrix
# -------------------------------------------------------

def save_confusion_matrix(

        labels,

        predictions,

        classes,

        experiment

):

    create_result_directories()

    cm = confusion_matrix(labels,predictions)

    disp = ConfusionMatrixDisplay(

        confusion_matrix=cm,

        display_labels=classes

    )

    fig, ax = plt.subplots(figsize=(8,8))

    disp.plot(

        cmap="Blues",

        ax=ax,

        colorbar=False

    )

    plt.title(f"{experiment} Confusion Matrix")

    filename = FIGURE_DIR / f"{experiment}_ConfusionMatrix.png"

    plt.savefig(filename,dpi=300)

    plt.close()

    print("✓ Confusion Matrix saved")

    return cm

# -------------------------------------------------------
# Classification Report
# -------------------------------------------------------

def save_classification_report(

        labels,

        predictions,

        experiment

):

    create_result_directories()

    report = classification_report(

        labels,

        predictions,

        digits=4

    )

    filename = REPORT_DIR / f"{experiment}_ClassificationReport.txt"

    with open(filename,"w") as f:

        f.write(report)

    print("✓ Classification Report saved")

    return report

# -------------------------------------------------------
# Precision Recall F1
# -------------------------------------------------------

def calculate_metrics(

        labels,

        predictions

):

    precision = precision_score(

        labels,

        predictions,

        average="macro"

    )

    recall = recall_score(

        labels,

        predictions,

        average="macro"

    )

    f1 = f1_score(

        labels,

        predictions,

        average="macro"

    )

    return precision, recall, f1

# -------------------------------------------------------
# Comparison Plot
# -------------------------------------------------------

def plot_comparison_curves(

        histories,

        metric,

        title,

        filename

):

    """
    histories

    {

      "FedAvg":history1,

      "DP1":history2,

      ...

    }

    """

    plt.figure(figsize=(9,6))

    for name, history in histories.items():

        plt.plot(

            history["round"],

            history[metric],

            linewidth=2,

            marker="o",

            markersize=3,

            label=name

        )

    plt.xlabel("Communication Round")

    plt.ylabel(metric.replace("_"," ").title())

    plt.title(title)

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        FIGURE_DIR / filename,

        dpi=300

    )

    plt.close()

    print("✓ Comparison figure saved")

# -------------------------------------------------------
# Save All Figures
# -------------------------------------------------------

def save_all_figures(

        history,

        labels,

        predictions,

        class_names,

        experiment

):

    plot_accuracy_curve(

        history,

        experiment

    )

    plot_loss_curve(

        history,

        experiment

    )

    save_confusion_matrix(

        labels,

        predictions,

        class_names,

        experiment

    )

    save_classification_report(

        labels,

        predictions,

        experiment

    )

    print("✓ All figures generated")


# -------------------------------------------------------
# Generate Complete Experiment Report
# -------------------------------------------------------

def generate_full_report(

        history,

        labels,

        predictions,

        class_names,

        experiment,

        method,

        clients,

        rounds,

        local_epochs,

        learning_rate,

        batch_size,

        clip_norm,

        noise,

        best_accuracy,

        final_accuracy,

        training_time

):
    """
    Generates every report, table and figure
    for one Federated Learning experiment.
    """

    print("\nGenerating Experiment Report...")

    # ---------------------------------------
    # Metrics
    # ---------------------------------------

    precision, recall, f1 = calculate_metrics(

        labels,

        predictions

    )

    # ---------------------------------------
    # Save History CSV
    # ---------------------------------------

    save_round_history(

        history,

        experiment

    )

    # ---------------------------------------
    # Save Figures
    # ---------------------------------------

    save_all_figures(

        history,

        labels,

        predictions,

        class_names,

        experiment

    )

    # ---------------------------------------
    # Build Summary
    # ---------------------------------------

    summary = build_summary(

        experiment=experiment,

        method=method,

        clients=clients,

        rounds=rounds,

        local_epochs=local_epochs,

        learning_rate=learning_rate,

        batch_size=batch_size,

        clip_norm=clip_norm,

        noise=noise,

        best_accuracy=best_accuracy,

        final_accuracy=final_accuracy,

        precision=precision,

        recall=recall,

        f1=f1,

        training_time=training_time

    )

    # ---------------------------------------
    # Reports
    # ---------------------------------------

    save_experiment_summary(

        summary,

        experiment

    )

    save_metadata(

        summary,

        experiment

    )

    # ---------------------------------------
    # Experiment Log
    # ---------------------------------------

    update_experiment_log(

        experiment,

        method,

        clients,

        rounds,

        local_epochs,

        learning_rate,

        batch_size,

        clip_norm,

        noise,

        best_accuracy,

        final_accuracy,

        precision,

        recall,

        f1,

        training_time

    )

    print("="*60)

    print("Experiment reporting completed successfully.")

    print("="*60)

    return summary

# -------------------------------------------------------
# Load History CSV
# -------------------------------------------------------

def load_history(experiment):

    filename = HISTORY_DIR / f"{experiment}_history.csv"

    if not filename.exists():

        raise FileNotFoundError(filename)

    return pd.read_csv(filename)

# -------------------------------------------------------
# Generate Comparison Figures
# -------------------------------------------------------

def generate_comparison_figures(experiments):
    """
    experiments

    [

      "FedAvg",

      "DP_Validation",

      "DP1",

      "DP2"

    ]
    """

    histories = {}

    for exp in experiments:

        histories[exp] = load_history(exp)

    plot_comparison_curves(

        histories,

        metric="accuracy",

        title="Accuracy Comparison",

        filename="Comparison_Accuracy.png"

    )

    plot_comparison_curves(

        histories,

        metric="train_loss",

        title="Training Loss Comparison",

        filename="Comparison_TrainLoss.png"

    )

    plot_comparison_curves(

        histories,

        metric="test_loss",

        title="Test Loss Comparison",

        filename="Comparison_TestLoss.png"

    )

    print("Comparison figures generated.")

if __name__ == "__main__":

    create_result_directories()

    print("="*60)

    print("REPORTING MODULE READY")

    print("="*60)