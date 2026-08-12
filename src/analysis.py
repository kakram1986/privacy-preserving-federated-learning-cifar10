"""
=========================================================
Privacy-Preserving Federated Learning Framework

Module : Analysis
Version: 3.0 (Final Dissertation Release)

Author : Khawar Akram
=========================================================
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import config

# ---------------------------------------------------------
# Directories
# ---------------------------------------------------------

RESULTS_DIR = Path(config.RESULTS_PATH)

TABLE_DIR = RESULTS_DIR / "Tables"
FIGURE_DIR = RESULTS_DIR / "Figures"

ANALYSIS_DIR = RESULTS_DIR / "Analysis"
ANALYSIS_TABLES = ANALYSIS_DIR / "Tables"
ANALYSIS_FIGURES = ANALYSIS_DIR / "Figures"

LOG_FILE = TABLE_DIR / "federated_experiment_log.csv"
HISTORY_DIR = TABLE_DIR / "Round_History"


# ---------------------------------------------------------
# Create Directories
# ---------------------------------------------------------

def create_analysis_directories():

    ANALYSIS_TABLES.mkdir(parents=True, exist_ok=True)
    ANALYSIS_FIGURES.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Experiment Groups
# ---------------------------------------------------------

BASELINE_EXPERIMENTS = ["FedAvg"]

VALIDATION_EXPERIMENTS = [
    "DP_Validation",
    "DP-0R",
]

FINAL_EXPERIMENTS = [
    "FedAvg",
    "DP-1",
    "DP-2",
    "DP-3",
    "DP-4",
    "SA-1",
]

# ---------------------------------------------------------
# Load Experiment Log
# ---------------------------------------------------------

def load_experiment_log():

    if not LOG_FILE.exists():
        raise FileNotFoundError(LOG_FILE)

    log = pd.read_csv(LOG_FILE)

    # Keep only the latest occurrence of each experiment
    log = log.drop_duplicates(
        subset="Experiment",
        keep="last"
    ).reset_index(drop=True)

    print("=" * 60)
    print("Experiment Log Loaded")
    print("=" * 60)
    print(f"Experiments : {len(log)}")
    print()

    return log


# ---------------------------------------------------------
# Load History
# ---------------------------------------------------------

def load_history(experiment):

    filename = HISTORY_DIR / f"{experiment}_history.csv"

    if not filename.exists():
        print(f"History not found : {experiment}")
        return None

    return pd.read_csv(filename)


# ---------------------------------------------------------
# Validate Experiments
# ---------------------------------------------------------

def validate_experiments(log):

    print("=" * 60)
    print("Checking Experiment Files")
    print("=" * 60)

    valid = []

    for experiment in log["Experiment"]:

        history_file = HISTORY_DIR / f"{experiment}_history.csv"

        if history_file.exists():

            print(f"✓ {experiment}")

            valid.append(experiment)

        else:

            print(f"✗ Missing History : {experiment}")

    print()

    print(f"Valid Experiments : {len(valid)}")

    return valid


# ---------------------------------------------------------
# Load All Histories
# ---------------------------------------------------------

def load_all_histories(log):

    histories = {}

    for experiment in log["Experiment"]:

        history = load_history(experiment)

        if history is not None:

            histories[experiment] = history

    return histories

# ---------------------------------------------------------
# Dissertation Experiments
# ---------------------------------------------------------

def dissertation_log(log):

    return log[
        log["Experiment"].isin(FINAL_EXPERIMENTS)
    ].reset_index(drop=True)


# ---------------------------------------------------------
# Sort Experiments
# ---------------------------------------------------------

def sort_experiments(log):

    order = {

        "FedAvg":0,

        "DP_Validation":1,

        "DP-0R":2,

        "DP-1":3,

        "DP-2":4,

        "DP-3":5,

        "DP-4":6,

        "SA-1":7,

    }

    log["Order"] = log["Experiment"].map(order)

    log = log.sort_values("Order")

    log = log.drop(columns="Order")

    return log.reset_index(drop=True)

# ---------------------------------------------------------
# Global Figure Style
# ---------------------------------------------------------

COLORS = {
    "FedAvg": "#1f77b4",
    "DP-1": "#2ca02c",
    "DP-2": "#ff7f0e",
    "DP-3": "#d62728",
    "DP-4": "#9467bd",
    "SA-1": "#8c564b",
}

def apply_figure_style():

    plt.rcParams.update({

        "figure.figsize": (8,5),

        "figure.dpi":300,

        "font.size":11,

        "axes.titlesize":13,

        "axes.labelsize":11,

        "legend.fontsize":10,

        "xtick.labelsize":10,

        "ytick.labelsize":10,

        "axes.grid":True,

        "grid.alpha":0.30,

    })


def save_figure(filename):

    plt.tight_layout()

    plt.savefig(

        ANALYSIS_FIGURES / filename,

        dpi=300,

        bbox_inches="tight",

    )

    plt.close()

    print(f"✓ {filename}")


# ---------------------------------------------------------
# Generic Metric Plot
# ---------------------------------------------------------

def plot_metric(
    histories,
    metric,
    ylabel,
    title,
    filename,
):
    apply_figure_style()

    plt.figure(figsize=(9, 6))

    for experiment, history in histories.items():

        if metric not in history.columns:
            continue

        color = COLORS.get(experiment, None)

        plt.plot(

            history["round"],

            history[metric],

            linewidth=2.5,

            color=color,

            label=experiment,

        )

    plt.xlabel("Communication Round")

    plt.ylabel(ylabel)

    plt.title(title)

    plt.grid(alpha=0.30)

    plt.legend(frameon=True)

    plt.tight_layout()

    plt.savefig(

        ANALYSIS_FIGURES / filename,

        dpi=300,

        bbox_inches="tight",

    )

    plt.close()

    print(f"✓ {filename}")

# ---------------------------------------------------------
# Accuracy Comparison
# ---------------------------------------------------------

def plot_accuracy_comparison(histories):

    plot_metric(

        histories=histories,

        metric="accuracy",

        ylabel="Accuracy (%)",

        title="Accuracy Comparison",

        filename="Figure_4_1_AccuracyComparison.png",

    )


# ---------------------------------------------------------
# Training Loss Comparison
# ---------------------------------------------------------

def plot_training_loss(histories):

    plot_metric(

        histories=histories,

        metric="train_loss",

        ylabel="Training Loss",

        title="Training Loss Comparison",

        filename="Figure_4_2_TrainingLossComparison.png",

    )


# ---------------------------------------------------------
# Test Loss Comparison
# ---------------------------------------------------------

def plot_test_loss(histories):
    
    plot_metric(

        histories=histories,

        metric="test_loss",

        ylabel="Test Loss",

        title="Test Loss Comparison",

        filename="Figure_4_3_TestLossComparison.png",

    )


# ---------------------------------------------------------
# Save Summary Table
# ---------------------------------------------------------

def save_summary_table(log):

    columns = [

        "Experiment",

        "Method",

        "NoiseMultiplier",

        "ClipNorm",

        "BestAccuracy",

        "TrainingTime(min)"

    ]

    log[columns].to_csv(

        ANALYSIS_TABLES /

        "Federated_Results_Summary.csv",

        index=False

    )


# ---------------------------------------------------------
# Best Accuracy Comparison
# ---------------------------------------------------------

def plot_best_accuracy(log):

    plt.figure(figsize=(8,5))

    colors = [
        COLORS.get(exp, "#808080")
        for exp in log["Experiment"]
    ]

    plt.bar(

        log["Experiment"],

        log["BestAccuracy"],

        color=colors,

        edgecolor="black"

    )

    # Write values above bars
    for i, value in enumerate(log["BestAccuracy"]):

        plt.text(

            i,

            value + 0.5,

            f"{value:.2f}",

            ha="center",

            fontsize=10,

        )

    plt.ylabel("Best Accuracy (%)")

    plt.xlabel("Experiment")

    plt.title("Best Accuracy Comparison")

    plt.grid(axis="y", alpha=0.30)

    save_figure(
        "Figure_4_4_BestAccuracyComparison.png"
    )

    print("✓ Figure_4_4_BestAccuracyComparison.png")

# ---------------------------------------------------------
# Privacy Utility Trade-off
# ---------------------------------------------------------

def plot_privacy_tradeoff(log):

    dp_log = log[

        log["Experiment"].str.contains("DP")

    ].copy()

    plt.figure(figsize=(8,5))

    plt.plot(

        dp_log["NoiseMultiplier"],

        dp_log["BestAccuracy"],

        marker="o",

        linewidth=2.5,

        markersize=8,

        color="#d62728",

    )

    for _, row in dp_log.iterrows():

        plt.text(

            row["NoiseMultiplier"],

            row["BestAccuracy"] + 0.6,

            row["Experiment"],

            fontsize=9,

            ha="center",

        )

    plt.xlabel("Noise Multiplier")

    plt.ylabel("Best Accuracy (%)")

    plt.title("Privacy–Utility Trade-off")

    plt.grid(alpha=0.30)

    save_figure(
        "Figure_4_5_PrivacyUtilityTradeoff.png"
    )

    print("✓ Figure_4_5_PrivacyUtilityTradeoff.png")

# ---------------------------------------------------------
# Training Time Comparison
# ---------------------------------------------------------

def plot_training_time(log):

    plt.figure(figsize=(8,5))

    colors = [

        COLORS.get(exp, "#808080")

        for exp in log["Experiment"]

    ]

    plt.bar(

        log["Experiment"],

        log["TrainingTime(min)"],

        color=colors,

        edgecolor="black",

    )

    for i, value in enumerate(log["TrainingTime(min)"]):

        plt.text(

            i,

            value + 0.05,

            f"{value:.2f}",

            ha="center",

            fontsize=9,

        )

    plt.ylabel("Training Time (minutes)")

    plt.xlabel("Experiment")

    plt.title("Training Time Comparison")

    plt.grid(axis="y", alpha=0.30)

    save_figure(
        "Figure_4_6_TrainingTimeComparison.png"
    )

    print("✓ Figure_4_6_TrainingTimeComparison.png")


# ---------------------------------------------------------
# Table 4.1
# Experimental Configuration
# ---------------------------------------------------------

def save_experiment_configuration(log):

    columns = [

        "Experiment",

        "Method",

        "Clients",

        "Rounds",

        "LocalEpochs",

        "LearningRate",

        "BatchSize",

        "ClipNorm",

        "NoiseMultiplier"

    ]

    table = log[columns].copy()

    filename = (

        ANALYSIS_TABLES /

        "Table_4_1_ExperimentalConfiguration.csv"

    )

    table.to_csv(

        filename,

        index=False

    )

    print("✓ Table_4_1_ExperimentalConfiguration.csv")

# ---------------------------------------------------------
# Table 4.2
# Overall Results Summary
# ---------------------------------------------------------

def save_results_summary(log):

    columns = [

        "Experiment",

        "Method",

        "NoiseMultiplier",

        "BestAccuracy",

        "FinalAccuracy",

        "TrainingTime(min)"

    ]

    table = log[columns].copy()

    filename = (

        ANALYSIS_TABLES /

        "Table_4_2_ResultsSummary.csv"

    )

    table.to_csv(

        filename,

        index=False

    )

    print("✓ Table_4_2_ResultsSummary.csv")

# ---------------------------------------------------------
# Table 4.3
# Privacy Utility Trade-off
# ---------------------------------------------------------

def save_privacy_tradeoff(log):

    columns = [

        "Experiment",

        "NoiseMultiplier",

        "ClipNorm",

        "BestAccuracy"

    ]

    table = log[columns].copy()

    filename = (

        ANALYSIS_TABLES /

        "Table_4_3_PrivacyUtilityTradeoff.csv"

    )

    table.to_csv(

        filename,

        index=False

    )

    print("✓ Table_4_3_PrivacyUtilityTradeoff.csv")

# ---------------------------------------------------------
# Table 4.4
# Performance Metrics
# ---------------------------------------------------------

def save_performance_metrics(log):

    columns = [

        "Experiment",

        "BestAccuracy",

        "FinalAccuracy",

        "TrainingTime(min)"

    ]

    table = log[columns].copy()

    filename = (

        ANALYSIS_TABLES /

        "Table_4_4_PerformanceMetrics.csv"

    )

    table.to_csv(

        filename,

        index=False

    )

    print("✓ Table_4_4_PerformanceMetrics.csv")

# ---------------------------------------------------------
# Table 4.5
# Experiment Ranking
# ---------------------------------------------------------

def save_experiment_ranking(log):

    ranking = (

        log

        .sort_values(

            "BestAccuracy",

            ascending=False

        )

        .reset_index(drop=True)

    )

    ranking.insert(

        0,

        "Rank",

        range(

            1,

            len(ranking)+1

        )

    )

    filename = (

        ANALYSIS_TABLES /

        "Table_4_5_ExperimentRanking.csv"

    )

    ranking.to_csv(

        filename,

        index=False

    )

    print("✓ Table_4_5_ExperimentRanking.csv")

# ---------------------------------------------------------
# Table 4.6
# Comparative Performance Analysis
# ---------------------------------------------------------

def save_comparative_analysis(log):

    table = log.copy()

    # -------------------------------------------------
    # FedAvg baseline
    # -------------------------------------------------

    baseline = float(
        table.loc[
            table["Experiment"] == "FedAvg",
            "BestAccuracy"
        ].iloc[0]
    )

    # -------------------------------------------------
    # Accuracy Drop
    # -------------------------------------------------

    table["AccuracyDrop(%)"] = (

        baseline - table["BestAccuracy"]

    ).round(2)

    # -------------------------------------------------
    # Privacy Level
    # -------------------------------------------------

    def privacy_level(noise):

        if noise == 0:
            return "None"

        elif noise <= 0.001:
            return "Very Low"

        elif noise <= 0.005:
            return "Low"

        elif noise <= 0.010:
            return "Medium"

        else:
            return "High"

    table["PrivacyLevel"] = table[
        "NoiseMultiplier"
    ].apply(privacy_level)

    columns = [

        "Experiment",

        "PrivacyLevel",

        "NoiseMultiplier",

        "BestAccuracy",

        "AccuracyDrop(%)",

        "TrainingTime(min)"

    ]

    filename = (

        ANALYSIS_TABLES /

        "Table_4_6_ComparativePerformance.csv"

    )

    table[columns].to_csv(

        filename,

        index=False

    )

    print("✓ Table_4_6_ComparativePerformance.csv")

# ---------------------------------------------------------
# Statistical Summary
# ---------------------------------------------------------

def save_statistics(log):

    summary = {

        "Experiments":

            len(log),

        "Average Accuracy":

            round(

                log["BestAccuracy"].mean(),

                2,

            ),

        "Maximum Accuracy":

            round(

                log["BestAccuracy"].max(),

                2,

            ),

        "Minimum Accuracy":

            round(

                log["BestAccuracy"].min(),

                2,

            ),

        "Accuracy Std":

            round(

                log["BestAccuracy"].std(),

                2,

            ),

        "Average Training Time":

            round(

                log["TrainingTime(min)"].mean(),

                2,

            ),

    }

    df = pd.DataFrame(

        summary.items(),

        columns=["Metric","Value"]

    )

    filename = (

        ANALYSIS_TABLES /

        "Table_4_7_Statistics.csv"

    )

    df.to_csv(

        filename,

        index=False,

    )

    print("✓ Table_4_7_Statistics.csv")

# ---------------------------------------------------------
# Analysis Report
# ---------------------------------------------------------

def generate_analysis_report(log):

    baseline = float(

        log.loc[

            log["Experiment"]=="FedAvg",

            "BestAccuracy"

        ].iloc[0]

    )

    best = log.loc[

        log["BestAccuracy"].idxmax()

    ]

    worst = log.loc[

        log["BestAccuracy"].idxmin()

    ]

    avg_accuracy = log["BestAccuracy"].mean()

    report = []

    report.append("="*60)

    report.append("CHAPTER 4 RESULTS SUMMARY")

    report.append("="*60)

    report.append("")

    report.append(

        f"Experiments Analysed : {len(log)}"

    )

    report.append("")

    report.append(

        f"Baseline Accuracy : {baseline:.2f}%"

    )

    report.append("")

    report.append(

        f"Highest Accuracy : "

        f"{best['Experiment']} "

        f"({best['BestAccuracy']:.2f}%)"

    )

    report.append(

        f"Lowest Accuracy : "

        f"{worst['Experiment']} "

        f"({worst['BestAccuracy']:.2f}%)"

    )

    report.append("")

    report.append(

        f"Average Accuracy : "

        f"{avg_accuracy:.2f}%"

    )

    report.append("")

    report.append("Privacy–Utility Observation")

    report.append("---------------------------")

    report.append(

        "Increasing the Differential "

        "Privacy noise multiplier "

        "generally reduced model "

        "accuracy while improving "

        "privacy protection."

    )

    report.append("")

    report.append("="*60)

    filename = (

        ANALYSIS_DIR /

        "Analysis_Report.txt"

    )

    with open(

        filename,

        "w",

        encoding="utf-8"

    ) as f:

        f.write("\n".join(report))

    print("✓ Analysis_Report.txt")

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def generate_all_analysis():

    print("=" * 60)
    print("GENERATING ANALYSIS")
    print("=" * 60)

    # -----------------------------------------------------
    # Create Output Directories
    # -----------------------------------------------------

    create_analysis_directories()

    # -----------------------------------------------------
    # Load Experiment Log
    # -----------------------------------------------------

    log = load_experiment_log()

    # -----------------------------------------------------
    # Sort Experiments
    # -----------------------------------------------------

    log = sort_experiments(log)

    # -----------------------------------------------------
    # Validate History Files
    # -----------------------------------------------------

    validate_experiments(log)

    # -----------------------------------------------------
    # Select Dissertation Experiments
    # -----------------------------------------------------

    dissertation = dissertation_log(log)

    print("\nExperiments used for dissertation:")

    for exp in dissertation["Experiment"]:
        print(f"   • {exp}")

    print()

    # -----------------------------------------------------
    # Load Histories
    # -----------------------------------------------------

    histories = load_all_histories(dissertation)

    # -----------------------------------------------------
    # Existing Figures
    # -----------------------------------------------------

    plot_accuracy_comparison(histories)

    plot_training_loss(histories)

    plot_test_loss(histories)

    plot_best_accuracy(dissertation)

    plot_privacy_tradeoff(dissertation)

    plot_training_time(dissertation)

    # -----------------------------------------------------
    # Summary Table
    # -----------------------------------------------------

    save_summary_table(dissertation)

    # -----------------------------------------------------
    # Dissertation Tables
    # -----------------------------------------------------

    save_experiment_configuration(dissertation)

    save_results_summary(dissertation)

    save_privacy_tradeoff(dissertation)

    save_performance_metrics(dissertation)

    save_experiment_ranking(dissertation)

    # -----------------------------------------------------
    # Statistical Analysis
    # -----------------------------------------------------

    save_comparative_analysis(dissertation)

    save_statistics(dissertation)

    generate_analysis_report(dissertation)

    print("=" * 60)
    print("Analysis completed successfully.")
    print("=" * 60)


# ---------------------------------------------------------

if __name__ == "__main__":

    generate_all_analysis()