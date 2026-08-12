"""
=========================================================
Privacy-Preserving Federated Learning Framework

Module : Server
Version: 3.0 (Final Dissertation Release)

Author : Khawar Akram
=========================================================
"""

# ==========================================================
# FEDERATED SERVER
# ==========================================================

import model
import evaluation


class FederatedServer:
    """
    Federated Learning Server.

    Responsibilities:
        • Maintain the global model
        • Distribute global weights
        • Receive aggregated weights
        • Evaluate the global model
    """

    def __init__(self, device):

        self.device = device

        self.global_model = model.get_model().to(device)

    # ======================================================
    # GET GLOBAL MODEL
    # ======================================================

    def get_model(self):
        """
        Returns the global model.
        """
        return self.global_model

    # ======================================================
    # GET GLOBAL WEIGHTS
    # ======================================================

    def get_weights(self):
        """
        Returns global model parameters.
        """
        return self.global_model.state_dict()

    # ======================================================
    # UPDATE GLOBAL WEIGHTS
    # ======================================================

    def set_weights(self, global_weights):
        """
        Update global model using aggregated weights.
        """
        self.global_model.load_state_dict(global_weights)

    # ======================================================
    # EVALUATE GLOBAL MODEL
    # ======================================================

    def evaluate(
        self,
        test_loader,
        criterion,
    ):
        """
        Evaluate the current global model.
        """

        return evaluation.evaluate(
            self.global_model,
            test_loader,
            criterion,
            self.device,
        )