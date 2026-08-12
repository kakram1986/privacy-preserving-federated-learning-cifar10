"""
=========================================================
Privacy-Preserving Federated Learning Framework

Module : Privacy
Version: 3.0 (Final Dissertation Release)

Author : Khawar Akram
=========================================================
"""

# ==========================================================
# DIFFERENTIAL PRIVACY
# ==========================================================

import copy
import torch


# ==========================================================
# TRAINABLE PARAMETER CHECK
# ==========================================================

def is_trainable_parameter(key, tensor):
    """
    Returns True only for trainable parameters.

    Excludes BatchNorm running statistics and counters.
    """

    if not tensor.dtype.is_floating_point:
        return False

    excluded = (
        "running_mean",
        "running_var",
        "num_batches_tracked",
    )

    return not any(x in key for x in excluded)


# ==========================================================
# COMPUTE MODEL UPDATE
# ==========================================================

def compute_model_update(global_model, local_model):
    """
    Compute client model update.

    ΔW = W_local - W_global
    """

    global_state = global_model.state_dict()
    local_state = local_model.state_dict()

    update = {}

    for key in global_state:

        if is_trainable_parameter(key, global_state[key]):

            update[key] = (
                local_state[key]
                - global_state[key]
            ).clone()

        else:

            # Buffers are copied but NOT privatized
            update[key] = global_state[key].clone()

    return update


# ==========================================================
# COMPUTE GLOBAL UPDATE NORM
# ==========================================================

def compute_update_norm(update):

    total_norm_sq = None

    for key, tensor in update.items():

        if not is_trainable_parameter(key, tensor):
            continue

        value = tensor.pow(2).sum()

        if total_norm_sq is None:
            total_norm_sq = value
        else:
            total_norm_sq += value

    if total_norm_sq is None:
        return 0.0

    return torch.sqrt(total_norm_sq).item()


# ==========================================================
# CLIP MODEL UPDATE
# ==========================================================

def clip_model_update(
    update,
    max_norm,
):
    """
    Clip client update using global L2 norm.
    """

    norm = compute_update_norm(update)

    if norm == 0:
        return update, 1.0

    clip_coef = min(
        1.0,
        max_norm / (norm + 1e-12),
    )

    if clip_coef < 1.0:

        for key in update:

            if is_trainable_parameter(key, update[key]):

                update[key].mul_(clip_coef)

    return update, clip_coef


# ==========================================================
# ADD GAUSSIAN NOISE
# ==========================================================

def add_gaussian_noise(
    update,
    noise_multiplier,
    max_norm,
):
    """
    Add Gaussian noise to clipped updates.
    """

    sigma = noise_multiplier * max_norm

    for key in update:

        if not is_trainable_parameter(key, update[key]):
            continue

        noise = (
            torch.randn_like(update[key])
            * sigma
        )

        update[key].add_(noise)

    return update


# ==========================================================
# RECONSTRUCT LOCAL MODEL
# ==========================================================

def reconstruct_local_model(
    global_model,
    local_model,
    noisy_update,
):
    """
    Reconstruct privatized local model.

    Trainable parameters:
        W = W_global + ΔW_private

    BatchNorm buffers:
        Preserved from the trained local model.
    """

    noisy_model = copy.deepcopy(global_model)

    global_state = global_model.state_dict()
    local_state = local_model.state_dict()

    new_state = {}

    for key in global_state:

        if is_trainable_parameter(key, global_state[key]):

            new_state[key] = (
                global_state[key]
                + noisy_update[key]
            )

        else:

            # Preserve BatchNorm running statistics
            new_state[key] = local_state[key].clone()

    noisy_model.load_state_dict(new_state)

    return noisy_model


# ==========================================================
# APPLY CLIENT DP
# ==========================================================

def apply_client_dp(
    global_model,
    local_model,
    max_norm,
    noise_multiplier,
):
    """
    Apply Client-Level Differential Privacy.
    """

    update = compute_model_update(
        global_model,
        local_model,
    )

    if DEBUG_DP:

        print(
            f"[DP] Update Norm Before Clip : "
            f"{compute_update_norm(update):.6f}"
        )

    update, clip_coef = clip_model_update(
        update,
        max_norm,
    )

    if DEBUG_DP:

        print(
            f"[DP] Update Norm After Clip  : "
            f"{compute_update_norm(update):.6f}"
        )

        print(
            f"[DP] Clip Coefficient        : "
            f"{clip_coef:.6f}"
        )

    update = add_gaussian_noise(
        update,
        noise_multiplier,
        max_norm,
    )

    if DEBUG_DP:

        print(
            f"[DP] Update Norm After Noise : "
            f"{compute_update_norm(update):.6f}"
        )

        print("-" * 60)

    noisy_model = reconstruct_local_model(
        global_model,
        local_model,
        update,
    )

    return noisy_model