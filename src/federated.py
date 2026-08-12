"""
=========================================================
Privacy-Preserving Federated Learning Framework

Module : Federated
Version: 3.0 (Final Dissertation Release)

Author : Khawar Akram
=========================================================
"""

# ==========================================================
# FEDERATED LEARNING ENGINE
# ==========================================================

import copy

import torch
import torch.nn as nn

import aggregation
import client
import config
import dataset
import server
import evaluation

from reporting import (
    start_timer,
    stop_timer,
    initialise_history,
    update_history,
    generate_full_report,
)


class FederatedLearning:
    """
    Federated Learning Engine.

    Coordinates the complete Federated Averaging (FedAvg)
    workflow.

    Responsibilities
    ----------------
    • Create Server
    • Create Clients
    • Execute Communication Rounds
    • Aggregate Client Models
    • Evaluate Global Model
    """

    # ======================================================
    # INITIALIZATION
    # ======================================================

    def __init__(self):

        self.device = config.DEVICE

        # ---------------------------------------------
        # Global Server
        # ---------------------------------------------

        self.server = server.FederatedServer(self.device)

        # ---------------------------------------------
        # Client List
        # ---------------------------------------------

        self.clients = []

        # ---------------------------------------------
        # Test Loader (Load Once)
        # ---------------------------------------------

        _, self.test_loader = dataset.get_dataloaders()

        # ---------------------------------------------
        # Loss Function
        # ---------------------------------------------

        self.criterion = nn.CrossEntropyLoss()

    # ======================================================
    # CREATE CLIENTS
    # ======================================================

    def create_clients(self):
        """
        Create all federated clients.
        """

        self.clients = []

        for client_id in range(config.NUM_CLIENTS):

            train_loader = dataset.get_client_loader(client_id)

            fl_client = client.FederatedClient(
                client_id=client_id,
                train_loader=train_loader,
                device=self.device,
            )

            self.clients.append(fl_client)

        print(f"✅ {len(self.clients)} clients created.")

    # ======================================================
    # ONE COMMUNICATION ROUND
    # ======================================================

    def run_round(self):
        """
        Execute one communication round.
        """

        # -------------------------------------------------
        # Global Model
        # -------------------------------------------------

        global_model = self.server.get_model()

        client_models = []
        client_samples = []
        client_masks = []
        client_histories = []

        # -------------------------------------------------
        # Local Client Training
        # -------------------------------------------------

        for fl_client in self.clients:

            (
                local_weights,
                local_mask,
                local_history,
                num_samples,
            ) = fl_client.train(
                global_model=global_model,
                criterion=self.criterion,
                optimizer_class=torch.optim.Adam,
                learning_rate=config.CLIENT_LEARNING_RATE,
                weight_decay=config.CLIENT_WEIGHT_DECAY,
                local_epochs=config.LOCAL_EPOCHS,
            )

            client_models.append(local_weights)
            client_masks.append(local_mask)
            client_samples.append(num_samples)
            client_histories.append(local_history)

        # -------------------------------------------------
        # Average Local Statistics
        # -------------------------------------------------

        if client_histories:

            average_train_loss = sum(
                h[-1]["loss"] for h in client_histories
            ) / len(client_histories)

            average_train_accuracy = sum(
                h[-1]["accuracy"] for h in client_histories
            ) / len(client_histories)

        else:

            average_train_loss = 0.0
            average_train_accuracy = 0.0

        # -------------------------------------------------
        # Optional Debug Validation
        # -------------------------------------------------

        if config.DEBUG:

            temp_model = copy.deepcopy(global_model)
            temp_model.load_state_dict(client_models[0])

            _, client_acc, _, _ = evaluation.evaluate(
                temp_model,
                self.test_loader,
                self.criterion,
                self.device,
            )

            print(
                f"Client 0 Accuracy : {client_acc:.2f}%"
            )


        # -------------------------------------------------
        # Global Aggregation
        # -------------------------------------------------

        if config.ENABLE_SECURE_AGGREGATION:

            global_weights = aggregation.secure_fedavg(

                client_models=client_models,

                client_masks=client_masks,

                client_samples=client_samples,

            )

        else:

            global_weights = aggregation.fedavg(

                client_models=client_models,

                client_samples=client_samples,

            )

        self.server.set_weights(global_weights)

        # -------------------------------------------------
        # Evaluate Global Model
        # -------------------------------------------------

        (
            test_loss,
            test_accuracy,
            predictions,
            targets,
        ) = self.server.evaluate(
            self.test_loader,
            self.criterion,
        )

        return {

            "train_loss": average_train_loss,

            "train_accuracy": average_train_accuracy,

            "test_loss": test_loss,

            "test_accuracy": test_accuracy,

            "predictions": predictions,

            "targets": targets,

        }

    # ======================================================
    # FEDERATED TRAINING
    # ======================================================

    def train(self):
        """
        Execute complete Federated Learning.

        Returns
        -------
        history
        best_accuracy
        predictions
        targets
        """

        # -----------------------------------------------------
        # Reporting Initialisation
        # -----------------------------------------------------

        experiment_timer = start_timer()
        history = initialise_history()

        best_accuracy = 0.0
        best_round = 0

        # Store the initial global model as a fallback
        best_weights = copy.deepcopy(
            self.server.get_weights()
        )

        predictions = None
        targets = None

        print("=" * 70)
        print("FEDERATED LEARNING")
        print("=" * 70)

        for rnd in range(config.ROUNDS):

            print(
                f"\nCommunication Round "
                f"{rnd + 1}/{config.ROUNDS}"
            )

            results = self.run_round()

            train_loss = results["train_loss"]
            train_accuracy = results["train_accuracy"]

            test_loss = results["test_loss"]
            test_accuracy = results["test_accuracy"]

            predictions = results["predictions"]
            targets = results["targets"]

            print(
                f"[Round {rnd+1:02d}/{config.ROUNDS}] "
                f"Train={train_loss:.4f} | "
                f"Test={test_loss:.4f} | "
                f"Acc={test_accuracy:.2f}%"
            )

            # -----------------------------------------
            # Save Best Global Model
            # -----------------------------------------

            if test_accuracy > best_accuracy:

                best_accuracy = test_accuracy
                best_round = rnd + 1

                best_weights = copy.deepcopy(
                    self.server.get_weights()
                )

                print(
                    f"★ New Best Accuracy: "
                    f"{best_accuracy:.2f}% "
                    f"(Round {best_round})"
                )

            update_history(

                history=history,

                round_number=rnd + 1,

                train_loss=train_loss,

                test_loss=test_loss,

                accuracy=test_accuracy,

                best_accuracy=best_accuracy

            )

        # ---------------------------------------------
        # Restore Best Global Model
        # ---------------------------------------------

        self.server.set_weights(best_weights)

        # --------------------------------------------------
        # Evaluate Restored Best Global Model
        # --------------------------------------------------

        (
            test_loss,
            test_accuracy,
            predictions,
            targets,
        ) = self.server.evaluate(
            self.test_loader,
            self.criterion,
        )

        training_time = stop_timer(experiment_timer)

        generate_full_report(

            history=history,

            labels=targets,

            predictions=predictions,

            class_names=config.CLASS_NAMES,

            experiment=config.EXPERIMENT_NAME,

            method=config.EXPERIMENT_METHOD,

            clients=config.NUM_CLIENTS,

            rounds=config.ROUNDS,

            local_epochs=config.LOCAL_EPOCHS,

            learning_rate=config.CLIENT_LEARNING_RATE,

            batch_size=config.BATCH_SIZE,

            clip_norm=config.MAX_GRAD_NORM,

            noise=config.NOISE_MULTIPLIER,

            best_accuracy=best_accuracy,

            final_accuracy=test_accuracy,

            training_time=training_time,

        )

        print("\n" + "=" * 70)
        print("FEDERATED TRAINING COMPLETE")
        print("=" * 70)
        print(f"Best Round      : {best_round}")
        print(f"Best Accuracy   : {best_accuracy:.2f}%")
        print(f"Training Time   : {training_time:.2f} min")
        print("=" * 70)

        return (
            history,
            best_accuracy,
            predictions,
            targets,
        )