"""
=========================================================
Privacy-Preserving Federated Learning Framework

Module : Client
Version: 3.0 (Final Dissertation Release)

Author : Khawar Akram
=========================================================
"""

# ==========================================================
# FEDERATED CLIENT
# ==========================================================

import copy

import config
import privacy

import secure_aggregation


class FederatedClient:

    def __init__(
        self,
        client_id,
        train_loader,
        device,
    ):

        self.client_id = client_id
        self.train_loader = train_loader
        self.device = device

    def train(
        self,
        global_model,
        criterion,
        optimizer_class,
        learning_rate,
        weight_decay,
        local_epochs,
    ):

        # Copy Global Model

        local_model = copy.deepcopy(global_model)
        local_model = local_model.to(self.device)

        optimizer = optimizer_class(
            local_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        history = []

        # -----------------------------------------
        # Local Training
        # -----------------------------------------

        for epoch in range(local_epochs):

            local_model.train()

            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in self.train_loader:

                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                optimizer.zero_grad()

                outputs = local_model(images)

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

                running_loss += loss.item() * labels.size(0)

                _, predicted = outputs.max(1)

                total += labels.size(0)

                correct += predicted.eq(labels).sum().item()

            epoch_loss = running_loss / total

            epoch_accuracy = 100 * correct / total

            history.append(
                {
                    "epoch": epoch + 1,
                    "loss": epoch_loss,
                    "accuracy": epoch_accuracy,
                }
            )

        # -----------------------------------------
        # Client-Level Differential Privacy
        # -----------------------------------------

        if config.ENABLE_DP:

            # Save original trained model
            original_state = copy.deepcopy(local_model.state_dict())

            # Apply Client-Level DP
            local_model = privacy.apply_client_dp(
                global_model=global_model,
                local_model=local_model,
                max_norm=config.MAX_GRAD_NORM,
                noise_multiplier=config.NOISE_MULTIPLIER,
            )

            # -----------------------------------------
            # Verify DP Reconstruction
            # -----------------------------------------

            max_diff = 0.0

            for key in original_state:

                if privacy.is_trainable_parameter(key, original_state[key]):

                    diff = (
                        original_state[key]
                        - local_model.state_dict()[key]
                    ).abs().max().item()

                    max_diff = max(max_diff, diff)

            if config.DEBUG:
                print(
                    f"[DP CHECK] Maximum Parameter Difference: "
                    f"{max_diff:.6f}"
                )

        # -----------------------------------------
        # Secure Aggregation
        # -----------------------------------------

        mask = None

        if config.ENABLE_SECURE_AGGREGATION:
            current_state = local_model.state_dict()

            mask = secure_aggregation.generate_mask(
                current_state
            )

            masked_state = secure_aggregation.apply_mask(
                current_state,
                mask,
            )

            local_model.load_state_dict(masked_state)

            if config.DEBUG:
                print(
                    f"[SA] Client {self.client_id} "
                    "model masked."
                )

        return (
            local_model.state_dict(),
            mask,
            history,
            len(self.train_loader.dataset),
        )