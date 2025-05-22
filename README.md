# Popcorn: GPU-Accelerated Kernel K-Means Clustering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) Popcorn is a high-performance implementation of the kernel K-means clustering algorithm, accelerated using GPUs. [cite: 6] It is designed to efficiently cluster datasets with complex, non-linearly separable structures, which classical K-means struggles with. Popcorn leverages sparse linear algebra techniques to achieve significant speedups compared to traditional CPU-based implementations.

This implementation is based on the work presented in the paper: "Popcorn: Accelerating kernel k-means on gpus through sparse linear algebra" by J. Bellavita, T. Pasquali, L. Del Rio Martin, F. Vella, and G. Guidi. 

## Overview

Kernel K-means extends the standard K-means algorithm by mapping data points into a higher-dimensional feature space via a kernel function. In this feature space, clusters that were originally non-linearly separable may become linearly separable, allowing for more accurate clustering. Popcorn focuses on an efficient, matrix-centric approach to kernel K-means, making it practical for moderately large datasets.

A key example of its effectiveness is clustering concentric circle datasets, where standard K-means fails.

## Key Features

* **GPU Acceleration:** Utilizes NVIDIA CUDA, cuBLAS, and cuSPARSE libraries for massive parallelism and optimized linear algebra operations.
* **Sparse Matrix Operations:** Employs a sparse representation (CSR format) for the cluster assignment matrix, enabling efficient Sparse Matrix-Matrix Multiply (SpMM) and Sparse Matrix-Vector Multiply (SpMV) operations.
* **Handles Non-Linear Clusters:** Effectively clusters data with non-linear boundaries (e.g., concentric circles) using kernel functions like the Gaussian RBF kernel.
* **Significant Performance Gains:** Achieves substantial speedups (over 1000x in tested scenarios, sometimes exceeding 10,000x) compared to CPU baselines.

## How Popcorn Works

1.  **Kernel Matrix Computation:** The kernel matrix $K$ (e.g., using Gaussian RBF kernel [cite: 16]) is precomputed on the GPU using cuBLAS.
    $\kappa(p_i, p_j) = \exp\left(-\frac{\|p_i - p_j\|^2}{2\sigma^2}\right)$
2.  **Sparse Cluster Assignment Matrix (V):** Cluster assignments are stored in a sparse matrix $V$ in CSR format.
3.  **Iterative Distance Computation & Assignment:**
    * The core distance components are computed using matrix operations:
        * $E = -2KV$ (Cross-term matrix, computed via SpMM). 
        * A vector $z$ is formed from $E$. 
        * Cluster norm vector $\tilde{C} = V^T z$ (computed via SpMV). 
    * The distance matrix $D = \tilde{P} + E + \tilde{C}$ (where $\tilde{P}$ contains self-similarities $K_{ii}$) is used to reassign points to the closest cluster. 
    * This process repeats until convergence. 
4.  **Optimized GPU Kernels:** Custom CUDA kernels are used for tasks like gathering cluster sums and parallel cluster assignment.

## Experimental Setup and Results (Example from CPE_810_Lab_04.pdf)

Popcorn was evaluated on a synthetic concentric circles dataset, a classic example of non-linearly separable clusters.

* **Dataset:** $n$ points in $\mathbb{R}^2$, forming two concentric circles. 
* **Kernel:** Gaussian RBF kernel with $\sigma=1.0$.
* **Comparison:** Against a C++ CPU baseline implementation of kernel K-means.

**Performance Highlights:**
* **Speedup:** Popcorn achieved over 1000x speedup at larger scales, with some configurations (e.g., $N=2000, K=5$ or $K=10$) showing speedups exceeding 10,000x and 12,000x respectively. [cite: 98, 100] (See Figure 1 in the PDF)
* **GFLOPs:** Showed increasing GFLOPs performance with larger matrix sizes (N), generally around 5-10 GFLOPs for $K=2$ with $N$ from 1000 to 10000. (See Figure 2 and Table 1 in the PDF)

## Requirements

* NVIDIA GPU with CUDA support
* CUDA Toolkit (nvcc compiler)
* cuBLAS library
* cuSPARSE library

## Compilation

To compile the Popcorn kernel K-means implementation, use the following command. Adjust the `-arch=sm_XX` flag according to your GPU's compute capability (e.g., `sm_70`, `sm_75`, `sm_86`).

```bash
nvcc popcorn_kernel_kmeans.cu -o popcorn_kmeans -arch=sm_60 -std=c++17 -lcublas -lcusparse -lineinfo -O3
```