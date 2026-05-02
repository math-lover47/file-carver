# Dataset Preparation and Optimization Report

This document outlines the systematic process of transforming the raw NapierOne-tiny dataset into optimized formats suitable for high-performance training with the Mamba architecture.

## 1. Source: NapierOne-tiny
The source corpus is a modern benchmark for file type identification, containing thousands of files across 100 classes. It includes both structured and high-entropy formats (compressed, encrypted).

## 2. Preprocessing Pipeline
To enable efficient training, I implemented a custom pipeline that performs the following:
*   **Recursive Extraction**: Automatically traverses nested archives to retrieve raw files.
*   **Block Slicing**: Slices files into fixed 512-byte blocks (matching standard disk sectors).
*   **AOT Compilation**: Utilizes Ahead-of-Time compilation for PyTorch extensions to accelerate data processing.

## 3. Dataset Versions

### NapierOne-Lean
A refined version of the original corpus:
*   **Pruning**: Removed classes with insufficient samples.
*   **Consolidation**: Grouped similar sub-formats into broader categories.
*   **Serialization**: Converted to compressed `.npy` arrays for fast memory-mapped access.

### NapierOne-Micro (Current Training Focus)
Specifically designed for rapid Proof-of-Concept (PoC) and training under hardware constraints (Kaggle T4 GPUs):
*   **Scope**: 5 diverse classes (JPG, PDF, PNG, MP3, TXT).
*   **Strategy**: File-level sampling (30 full files per class) instead of random block-level sampling.
*   **Integrity**: Maintains sequential byte coherence within each file, which is critical for Mamba's recurrent dynamics.

## 4. Key Benefits of the Optimization
*   **VRAM Efficiency**: Optimized loading allows training on limited hardware.
*   **Training Speed**: Reduction in training time from days to **~2-3 hours** on T4 GPUs.
*   **Convergence**: Improved stability of the `OneCycleLR` scheduler due to better class balance and lower noise in the Micro version.
