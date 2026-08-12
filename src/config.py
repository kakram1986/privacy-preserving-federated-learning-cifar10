"""
=========================================================
Privacy-Preserving Federated Learning Framework

Module : Config
Version: 3.0 (Final Dissertation Release)

Author : Khawar Akram
=========================================================
"""

# ==========================================================
# PROJECT CONFIGURATION
# ==========================================================

import os
import torch

PROJECT_ROOT = "/content/drive/MyDrive/COM748_Project"

DATASET_PATH = os.path.join(PROJECT_ROOT, "01_Dataset")
CHECKPOINTS_PATH = os.path.join(PROJECT_ROOT, "05_Checkpoints")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "06_Results")

TABLES_PATH = os.path.join(RESULTS_PATH, "Tables")
FIGURES_PATH = os.path.join(RESULTS_PATH, "Figures")
REPORTS_PATH = os.path.join(RESULTS_PATH, "Reports")

# ==========================================================
# CENTRALIZED TRAINING
# ==========================================================

BATCH_SIZE = 64

EPOCHS = 30

LEARNING_RATE = 0.001

WEIGHT_DECAY = 0.0

NUM_CLASSES = 10

# ==========================================================
# RANDOM SEED
# ==========================================================

import random
import numpy as np
import torch


SEED = 42


def set_seed(seed=SEED):
    """
    Set random seed for reproducibility.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# ==========================================================
# FEDERATED LEARNING
# ==========================================================

NUM_CLIENTS = 5

ROUNDS = 30

LOCAL_EPOCHS = 4      # was 2

CLIENT_LEARNING_RATE = 0.001

CLIENT_WEIGHT_DECAY = 1e-4    # was 0.0

NUM_WORKERS = 8

PIN_MEMORY = True

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==========================================================
# DIFFERENTIAL PRIVACY
# ==========================================================

ENABLE_DP = True

MAX_GRAD_NORM = 10.0

NOISE_MULTIPLIER = 0.001

DEBUG = False

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# ==========================================================
# SECURE AGGREGATION
# ==========================================================

ENABLE_SECURE_AGGREGATION = True


# ==========================================================
# Experiment Information
# ==========================================================

EXPERIMENT_NAME = "SA-1"

EXPERIMENT_METHOD = "FedAvg + Secure Aggregation"

# ==========================================================
# DEBUG FLAG
# ==========================================================

DEBUG_DP = False

# ---------------------------------------------------------
# Debug Configuration
# ---------------------------------------------------------

DEBUG_SECURE_AGGREGATION = False