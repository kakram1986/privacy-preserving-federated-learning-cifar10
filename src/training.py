"""
=========================================================
Privacy-Preserving Federated Learning Framework

Module : Training
Version: 3.0 (Final Dissertation Release)

Author : Khawar Akram
=========================================================
"""

# ==========================================================
# TRAINING UTILITIES
# ==========================================================

import torch
from tqdm import tqdm


# ==========================================================
# TRAIN ONE EPOCH
# ==========================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    """
    Train the model for one epoch.

    Returns:
        epoch_loss (float)
        epoch_accuracy (float)
    """

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        loader,
        desc="Training",
        leave=False
    )

    for images, labels in progress_bar:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix(
            Loss=f"{loss.item():.4f}"
        )

    epoch_loss = running_loss / len(loader)

    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy


# ==========================================================
# EVALUATE MODEL
# ==========================================================

def evaluate(
    model,
    loader,
    criterion,
    device,
):
    """
    Evaluate the model.

    Returns:
        epoch_loss
        epoch_accuracy
        predictions
        targets
    """

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    predictions = []
    targets = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

            predictions.extend(
                predicted.cpu().numpy()
            )

            targets.extend(
                labels.cpu().numpy()
            )

    epoch_loss = running_loss / len(loader)

    epoch_accuracy = 100 * correct / total

    return (
        epoch_loss,
        epoch_accuracy,
        predictions,
        targets,
    )