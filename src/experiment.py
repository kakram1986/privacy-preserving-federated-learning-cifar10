"""
=========================================================
Privacy-Preserving Federated Learning Framework

Module : Experiment
Version: 3.0 (Final Dissertation Release)

Author : Khawar Akram
=========================================================
"""

# ==========================================================
# EXPERIMENT MANAGEMENT
# ==========================================================

import os
import torch
import pandas as pd
from datetime import datetime

from config import (
    CHECKPOINTS_PATH,
    TABLES_PATH,
    REPORTS_PATH,
)

# ==========================================================
# FILE NAMES
# ==========================================================

def get_experiment_files(experiment):

    return {

        "checkpoint":
            f"BaselineCNN_{experiment['version']}_{experiment['id']}.pth",

        "history":
            f"{experiment['id']}_History.csv",

        "report":
            f"{experiment['id']}_Report.txt",

        "log":
            "experiment_log.csv",
    }


# ==========================================================
# SAVE CHECKPOINT
# ==========================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    history,
    best_accuracy,
    experiment,
):

    files = get_experiment_files(experiment)

    checkpoint_path = os.path.join(
        CHECKPOINTS_PATH,
        files["checkpoint"]
    )

    torch.save(
        {
            "experiment": experiment,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "best_accuracy": best_accuracy,
        },
        checkpoint_path,
    )

    print("✅ Checkpoint Saved")

    return checkpoint_path


# ==========================================================
# SAVE HISTORY
# ==========================================================

def save_history(history, experiment):

    files = get_experiment_files(experiment)

    history_path = os.path.join(
        TABLES_PATH,
        files["history"]
    )

    pd.DataFrame(history).to_csv(
        history_path,
        index=False,
    )

    print("✅ Training History Saved")


# ==========================================================
# UPDATE EXPERIMENT LOG
# ==========================================================

def update_experiment_log(
    experiment,
    best_accuracy,
    best_epoch,
    history,
    training_time,
):

    files = get_experiment_files(experiment)

    experiment_log_path = os.path.join(
        TABLES_PATH,
        files["log"]
    )

    row = {

        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),

        "Experiment ID": experiment["id"],
        "Version": experiment["version"],
        "Title": experiment["title"],
        "Description": experiment["description"],

        "CNN Architecture": experiment["cnn"],
        "Dataset": experiment["dataset"],

        "Epochs": experiment["epochs"],
        "Batch Size": experiment["batch_size"],
        "Learning Rate": experiment["learning_rate"],
        "Optimizer": experiment["optimizer"],
        "Scheduler": experiment["scheduler"],
        "Weight Decay": experiment["weight_decay"],

        "Data Augmentation": experiment["augmentation"],
        "Label Smoothing": experiment["label_smoothing"],
        "Early Stopping": experiment["early_stopping"],

        "Best Epoch": best_epoch,
        "Best Accuracy (%)": round(best_accuracy, 2),
        "Final Train Loss": round(history["train_loss"][-1], 4),
        "Final Test Loss": round(history["test_loss"][-1], 4),
        "Training Time (min)": round(training_time, 2),

        "Decision": "Accepted",
        "Notes": experiment["notes"],
    }

    log_df = pd.DataFrame([row])

    if os.path.exists(experiment_log_path):

        existing = pd.read_csv(experiment_log_path)

        log_df = pd.concat(
            [existing, log_df],
            ignore_index=True,
        )

    log_df.to_csv(
        experiment_log_path,
        index=False,
    )

    print("✅ Experiment Log Updated")


# ==========================================================
# GENERATE REPORT
# ==========================================================

def generate_report(
    experiment,
    best_accuracy,
    best_epoch,
    history,
    training_time,
):

    files = get_experiment_files(experiment)

    report_path = os.path.join(
        REPORTS_PATH,
        files["report"]
    )

    with open(report_path, "w") as f:

        f.write("=" * 70 + "\n")
        f.write(f"Experiment : {experiment['id']}\n")
        f.write("=" * 70 + "\n\n")

        for key, value in experiment.items():
            f.write(f"{key}: {value}\n")

        f.write("\n")

        f.write(f"Best Epoch        : {best_epoch}\n")
        f.write(f"Best Accuracy     : {best_accuracy:.2f}%\n")
        f.write(f"Training Time     : {training_time:.2f} min\n")
        f.write(f"Final Train Loss  : {history['train_loss'][-1]:.4f}\n")
        f.write(f"Final Test Loss   : {history['test_loss'][-1]:.4f}\n")

    print("✅ Report Generated")


# ==========================================================
# SUMMARY
# ==========================================================

def print_summary(
    experiment,
    best_accuracy,
    best_epoch,
):

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Experiment : {experiment['id']}")
    print(f"Best Epoch : {best_epoch}")
    print(f"Accuracy   : {best_accuracy:.2f}%")
    print("=" * 70)


# ==========================================================
# MASTER FUNCTION
# ==========================================================

def save_experiment(
    model,
    optimizer,
    scheduler,
    history,
    best_accuracy,
    best_epoch,
    training_time,
    experiment,
):

    save_checkpoint(
        model,
        optimizer,
        scheduler,
        history,
        best_accuracy,
        experiment,
    )

    save_history(
        history,
        experiment,
    )

    update_experiment_log(
        experiment,
        best_accuracy,
        best_epoch,
        history,
        training_time,
    )

    generate_report(
        experiment,
        best_accuracy,
        best_epoch,
        history,
        training_time,
    )

    print_summary(
        experiment,
        best_accuracy,
        best_epoch,
    )