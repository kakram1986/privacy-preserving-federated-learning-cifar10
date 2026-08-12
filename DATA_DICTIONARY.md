# Evidence and Data Dictionary

## Core tables

### `results/tables/Reconciled_Results_Summary.csv`

One row per registered experiment. It contains the configured noise setting, fixed endpoint, test-selected peak, peak round where available, macro metrics, runtime, and evidence status.

### `results/tables/Reconciled_Round_History.csv`

Per-round accuracy and loss evidence for FedAvg, DP-1, DP-3, DP-4, and SA-1. No intended DP-2 trajectory is included.

### `results/tables/Evidence_Register.csv`

Records the provenance and permitted analytical use of retained evidence. Consult this file before deriving new claims.

### `results/tables/Manuscript_Claims.csv`

Maps major reported claims to evidence sources and records any required qualification.

## Partition evidence

- `class_distribution.csv`: client-by-class counts.
- `iid_verification_summary.csv`: conservation and balance checks.
- `partition_metadata.json`: recorded split metadata.
- `partition_report.txt`: human-readable verification output.
- `experiment_log.csv` and `federated_experiment_log.csv`: retained execution registers; use the reconciled tables when values conflict.

## Class-level evidence

Confusion matrices and classification reports are retained for FedAvg, DP-1, DP-3, DP-4, and SA-1. They describe test-selected checkpoints. DP-2 class-level files are deliberately excluded because they belong to a conflicting later evidence bundle rather than the intended registered 0.005 run.

## Missing evidence

The archive does not contain the intended DP-2 per-round history or class predictions, repeated-seed results, a formal privacy-accounting output, network/communication measurements, energy measurements, cryptographic security tests, or an exact environment lockfile.

