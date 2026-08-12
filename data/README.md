# Dataset

## CIFAR-10

This project uses the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) to evaluate federated learning, client-update perturbation, and simulated mask-recovery aggregation.

CIFAR-10 contains 60,000 colour images of size 32 × 32 pixels across ten mutually exclusive classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The official dataset split contains:

| Split | Images |
|---|---:|
| Training | 50,000 |
| Test | 10,000 |
| Total | 60,000 |

## Dataset access

The complete CIFAR-10 dataset is not redistributed in this repository. It is downloaded directly from its recognised source using:

```python
torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True
)
```

The test set is obtained by setting `train=False`.

This approach avoids unnecessary duplication of the dataset and enables users to obtain a clean copy through the standard `torchvision` interface.

## Federated partitioning

The 50,000 training images are shuffled using random seed `42` and divided into five synthetic federated clients.

| Property | Executed setting |
|---|---:|
| Number of clients | 5 |
| Training images per client | 10,000 |
| Participation per round | 5 of 5 clients |
| Partitioning strategy | Approximately IID |
| Random seed | 42 |
| Global test images | 10,000 |
| Number of classes | 10 |

Each client contains examples from all ten CIFAR-10 classes. The official 10,000-image test set remains global and is used to evaluate the aggregated model.

The clients are simulated data partitions rather than real individuals or organisations. Consequently, the experiment evaluates controlled federated-learning behaviour and does not represent naturally heterogeneous user populations.

## Preprocessing

Training images undergo the following transformations:

1. Random cropping to 32 × 32 pixels after four-pixel padding.
2. Random horizontal flipping with probability `0.5`.
3. Conversion to PyTorch tensors.
4. Normalisation using the CIFAR-10 channel statistics defined in the project dataset module.

Test images are converted to tensors and normalised using the same channel statistics. Random augmentation is not applied to the test set.

## Retained dataset evidence

This repository retains metadata and verification evidence rather than a second copy of CIFAR-10.

Expected evidence files include:

```text
evidence/
├── class_distribution.csv
├── iid_verification_summary.csv
├── partition_metadata.json
└── partition_report.txt
```

These files document:

- the number of images assigned to each client;
- per-client class distributions;
- the partitioning seed and configuration;
- checks for missing or duplicated assignments;
- the degree of variation between client class counts.

## Reproducibility

To reproduce the dataset preparation:

1. Install the dependencies listed in the repository.
2. Allow `torchvision` to download CIFAR-10.
3. Use random seed `42`.
4. Apply the preprocessing configuration described above.
5. Create five approximately IID client partitions.
6. Verify that every client receives 10,000 training images.
7. Retain the official 10,000-image test set for global evaluation.

Example environment setup:

```bash
pip install torch torchvision numpy pandas
```

Run the repository’s dataset or experiment entry point according to the instructions in the main project README.

## Data integrity and limitations

The partitioning procedure provides a controlled basis for comparing the experimental conditions because the same dataset, client allocation, model, and training schedule are used throughout.

However, approximately IID partitions do not reproduce important real-world federated-learning conditions such as:

- non-IID client data;
- unequal client dataset sizes;
- partial client participation;
- client dropout;
- device and network heterogeneity;
- changing data distributions.

Results from this dataset configuration should therefore be interpreted as evidence from a controlled simulation rather than as proof of deployment-scale performance.

## Privacy and ethics

CIFAR-10 is a public machine-learning benchmark. The project does not collect personal data or associate images with simulated client identities.

Federated learning keeps the assigned image data within each simulated client’s local training process. Nevertheless, model updates can still contain information about local optimisation. This is why the project evaluates update perturbation while avoiding unsupported claims of complete or formally quantified privacy.

## Citation

If this repository or dataset configuration is used in academic work, cite the original CIFAR-10 source:

> A. Krizhevsky, “Learning Multiple Layers of Features from Tiny Images,” University of Toronto, Toronto, Canada, Technical Report, 2009.

Dataset website:  
https://www.cs.toronto.edu/~kriz/cifar.html

## Repository

Project repository:  
https://github.com/kakram1986/privacy-preserving-federated-learning-cifar10
