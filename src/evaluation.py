"""
=========================================================
Privacy-Preserving Federated Learning Framework

Module : Evaluation
Version: 3.0 (Final Dissertation Release)

Author : Khawar Akram
=========================================================
"""

# ==========================================================
# MODEL EVALUATION
# ==========================================================

import torch


def evaluate(
    model,
    loader,
    criterion,
    device,
):
    """
    Evaluate model performance.

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

            predictions.extend(predicted.cpu().numpy())

            targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader)

    epoch_accuracy = 100 * correct / total

    return (
        epoch_loss,
        epoch_accuracy,
        predictions,
        targets,
    )