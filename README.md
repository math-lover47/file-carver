# Mamba File Carver

A high-performance file fragment identification and carving system built on the **Mamba State Space Model (SSM)**. Developed during an industrial internship at **TSARKA (R&D Department)**.

## Project Overview
This project addresses the **File Fragment identification (FiFT)** problem: determining the file type of a raw 512-byte block without any metadata or headers. Unlike traditional CNN-based approaches, this system leverages the sequence-modeling power of Mamba to capture long-range byte dependencies.

## Model Architecture: MambaTriHead
The core of the project is the `MambaTriHead` architecture, designed for multi-task learning in digital forensics.

### Key Components:
*   **Backbone**: 12-layer Mamba State Space Model ($d_{model}=768$).
*   **Selective Scan**: Dynamically gates information within byte streams, allowing the model to focus on structural markers even in high-entropy data.
*   **Multi-Task Heads**:
    1.  **Classification Head**: Identifies file format (e.g., JPG, PDF).
    2.  **Boundary Head**: Detects file start/end markers.
    3.  **Re-ID Head**: Learns fragment embeddings using **Batch-Hard Triplet Loss** to facilitate file reassembly.

### Technical Stack:
*   **Deep Learning**: PyTorch, Mamba-SSM.
*   **Experiment Tracking**: Weights & Biases (W&B).
*   **Optimization**: Optuna (automated hyperparameter search).
*   **Infrastructure**: Kaggle T4 GPUs.

## Dataset
The model is trained on optimized subsets of the **NapierOne** corpus:
*   **Micro Dataset**: 5 classes, file-level sampling for sequential coherence.
*   **Lean Dataset**: 41 classes, serialized for fast access.
See [DATASET_REPORT.md](./DATASET_REPORT.md) for more details.

## Getting Started

### 1. Requirements
*   Python 3.10+
*   PyTorch 2.0+
*   `mamba-ssm` & `causal-conv1d`

### 2. Preprocessing
Run the following scripts to prepare your data:
```bash
python process_napier.py         # Extract and slice
python create_micro_dataset.py   # Create the training subset
```

### 3. Training
To start training on Kaggle/local GPU:
```bash
python train_mamba_kaggle.py
```

## Acknowledgments
This research was conducted under the supervision of **Igor Seniushin (Head of R&D, TSARKA)**. Special thanks to the TSARKA team for their guidance and for the industrial internship opportunity.