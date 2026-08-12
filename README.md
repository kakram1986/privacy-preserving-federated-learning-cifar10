# Privacy-Preserving Federated Learning on CIFAR-10

Research artefacts for a COM748 MSc project examining the utility effects of complete-client-update clipping and Gaussian perturbation in federated learning, together with an arithmetic mask-and-recovery aggregation simulation.

**Author:** Khawar Akram  
**Institution:** Ulster University  
**Contact:** akram-k@ulster.ac.uk

## Scope

The repository preserves the implementation and evidence used for six registered configurations:

| Run | Configured mechanism | Noise multiplier | Fixed endpoint accuracy |
|---|---|---:|---:|
| FedAvg | Baseline weighted federated averaging | 0 | 88.55% |
| DP-1 | Complete-update clipping and Gaussian noise | 0.001 | 88.80% |
| DP-2 | Same perturbation path | 0.005 | 85.08% |
| DP-3 | Same perturbation path | 0.010 | 63.57% |
| DP-4 | Same perturbation path | 0.020 | 20.92% |
| SA-1 | Perturbation at 0.001 plus mask recovery | 0.001 | 79.36% |

The matched experiment design used CIFAR-10, five IID synthetic clients, 30 communication rounds, four local epochs, full client participation, and a shared CNN.

## Important interpretation boundaries

- The perturbation code clips a complete trainable model update and adds Gaussian noise. No adjacency definition, client sampling, privacy composition, or epsilon-delta accountant was retained. The configured noise multipliers must not be interpreted as formal differential-privacy guarantees.
- SA-1 is an arithmetic mask-and-recovery simulation. The server receives masks, and the implementation has no key exchange, secret sharing, collusion threshold, dropout recovery, or cryptographic confidentiality guarantee.
- DP-2 is retained in the planned sequence because independent summary exports consistently record the intended 0.005 run and its 85.08% endpoint. Its intended round history and class predictions were not retained; consequently, DP-2 is excluded from trajectory, loss-curve, and confusion-matrix claims.
- Results come from one retained seed. No confidence intervals, significance tests, or repeated-seed superiority claims are supported.
- The best checkpoint was selected using repeated test-set evaluation. Fixed round-30 accuracy is therefore the principal endpoint; peak results and confusion matrices are descriptive, test-selected diagnostics.

## Repository structure

```text
.
|-- src/                         # Audited implementation snapshot
|-- notebooks/                   # Retained Colab orchestration notebooks
|-- results/
|   |-- tables/                  # Reconciled endpoints, histories, and claim register
|   `-- figures/                 # Architecture, performance, and class-level evidence
|-- evidence/
|   |-- partition/               # IID partition records and experiment registers
|   `-- limitations/             # DP-2 and evidence-boundary notes
|-- docs/Literature
|-- REPRODUCIBILITY.md
|-- DATA_DICTIONARY.md
|-- requirements.txt
|-- CITATION.cff
`-- LICENSE
```

## Quick start

The retained code is an audited Google Colab/Google Drive execution snapshot rather than a packaged command-line application. This preserves the implementation that generated the evidence.

1. Create a Python environment and install the listed dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Open `notebooks/01_Environment_Setup.ipynb` in Google Colab.
3. Copy `src/` into the project source directory used by the notebook.
4. In `src/config.py`, set `PROJECT_ROOT` to the relevant Google Drive project directory.
5. Review the requested run settings in `src/config.py`, especially `ENABLE_DP`, `NOISE_MULTIPLIER`, `ENABLE_SECURE_AGGREGATION`, `EXPERIMENT_NAME`, and `EXPERIMENT_METHOD`.
6. Execute `notebooks/Project_v2_FedAvg-DP_Final.ipynb` in cell order.

The notebooks should be reviewed before execution because the available environment record does not contain exact package versions. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the complete protocol and limitations.

## Evidence first

The most reliable starting points for reviewing the reported results are:

- `results/tables/Reconciled_Results_Summary.csv` for registered endpoints, peaks, and runtime;
- `results/tables/Reconciled_Round_History.csv` for runs with complete histories;
- `results/tables/Evidence_Register.csv` for the permitted use of each evidence source;
- `results/tables/Manuscript_Claims.csv` for source-to-claim controls;
- `evidence/limitations/DP2_Evidence_Note.txt` for the DP-2 qualification;
- `docs/Literature` for literature review evidence.

## Citation

Please use the metadata in `CITATION.cff`. A publication identifier may be added if the associated paper or repository archive receives one.

## Licence

Copyright (c) 2026 Khawar Akram. All rights reserved. See `LICENSE`.
