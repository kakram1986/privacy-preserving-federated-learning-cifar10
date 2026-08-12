# Reproducibility Guide

## Reproducibility status

This archive supports **method and evidence inspection** and **qualified computational reproduction**. It does not promise bit-for-bit recreation because exact package versions were not retained and the recorded CUDA/cuDNN execution was not deterministic.

## Retained execution design

- Dataset: CIFAR-10
- Clients: five synthetic IID clients
- Participation: all clients in every round
- Client size: 10,000 training samples each
- Test set: 10,000 samples
- Communication rounds: 30
- Local epochs: four
- Batch size: 64
- Optimiser: Adam
- Client learning rate: 0.001
- Client weight decay: 0.0001
- Seed: 42
- Accelerator behaviour: cuDNN benchmark enabled; deterministic execution disabled

## Suggested execution sequence

1. Use Google Colab with a CUDA-capable runtime if available.
2. Mount Google Drive and establish the project folders expected by `src/config.py`.
3. Install the dependencies in `requirements.txt`. Record the resolved versions with `python -m pip freeze` before training.
4. Run `notebooks/01_Environment_Setup.ipynb` and verify imports and paths.
5. Review the experiment switches in `src/config.py` before every run.
6. Execute `notebooks/Project_v2_FedAvg-DP_Final.ipynb` in cell order.
7. Retain the full history, metadata, classification report, predictions, checkpoint-selection record, and runtime for each run.
8. Compare new outputs with `results/tables/Reconciled_Results_Summary.csv`; do not overwrite the archived evidence.

## Planned experiment sequence

| Run | `ENABLE_DP` | `NOISE_MULTIPLIER` | `ENABLE_SECURE_AGGREGATION` | Interpretation |
|---|---:|---:|---:|---|
| FedAvg | False | 0 | False | Baseline |
| DP-1 | True | 0.001 | False | Low configured update-noise setting |
| DP-2 | True | 0.005 | False | Intermediate setting; archived evidence is summary-level only |
| DP-3 | True | 0.010 | False | Larger setting |
| DP-4 | True | 0.020 | False | Largest retained setting |
| SA-1 | True | 0.001 | True | Perturbation plus simulated mask recovery |

Set `EXPERIMENT_NAME` and `EXPERIMENT_METHOD` consistently with each run. The repository preserves the final SA-1-oriented configuration snapshot, so these fields must be reviewed rather than assumed.

## Validation checks

Before accepting a reproduced result, confirm:

1. Five client partitions contain 10,000 training examples each.
2. The grand training total is 50,000, with 5,000 samples per class.
3. All five clients participate in all 30 rounds.
4. FedAvg weights equal client sample counts divided by the total sample count.
5. Batch-normalisation buffers are excluded from the clipping-and-noise transformation.
6. The fixed round-30 endpoint is reported separately from the test-selected peak.
7. A confusion matrix is linked to the same retained run and checkpoint as its classification report.
8. Any absent or conflicting artefact is disclosed instead of reconstructed retrospectively.

## Recommended stronger replication

A publication-grade replication should introduce a validation split for checkpoint selection, run at least three deterministic seeds per condition, report uncertainty and paired comparisons, retain an immutable manifest, and add automated tests for aggregation, clipping, mask recovery, partition conservation, and report generation. Formal privacy claims additionally require a declared protected unit, sampling model, calibrated clipping, composition, and an epsilon-delta accountant.

