# Matrix Multiplication — Basics

This document explains matrix multiplication from first principles and how it maps to GPU execution.

---

## What Matrix Multiplication Is

Given two matrices:

- Matrix **A** of size `M × K`
- Matrix **B** of size `K × N`

Their product is matrix **C** of size `M × N`.

Matrix multiplication is defined as:

```
C[i][j] = sum over k of ( A[i][k] * B[k][j] )
```

Each element of the output matrix is computed as a **dot product** of:
- one row from **A**
- one column from **B**

---

## Example

Matrix A (2 × 3):

```
1  2  3
4  5  6
```

Matrix B (3 × 2):

```
7   8
9  10
11 12
```

Result C (2 × 2):

```
58   64
139 154
```

Each value in `C` is computed independently.

---

## Key Observation

Each output element `C[i][j]` is independent of all other elements.

This makes matrix multiplication ideal for parallel execution.

---

## CPU Implementation (Baseline)

On the CPU, matrix multiplication is typically written using three nested loops:

```cpp
for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        float sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[i*K + k] * B[k*N + j];
        }
        C[i*N + j] = sum;
    }
}
```

---

## How This Maps to CUDA

In CUDA:

- Threads replace the outer loops (`i` and `j`)
- The inner loop (`k`) remains inside the kernel

Each CUDA thread computes **one output element** `C[row][col]`.

---

## Thread Mapping

Each thread computes its position using:

```cpp
row = blockIdx.y * blockDim.y + threadIdx.y;
col = blockIdx.x * blockDim.x + threadIdx.x;
```

This maps threads directly to the output matrix `C`.

---

## Bounds Checking

More threads are often launched than output elements.

Each thread must check:

```cpp
if (row < M && col < N) {
    // safe to compute C[row][col]
}
```

This prevents invalid memory access.

---

## Memory Layout

Matrices are stored as **flattened 1D arrays** in row-major order.

Access patterns:

- `A[row][k]` → `A[row * K + k]`
- `B[k][col]` → `B[k * N + col]`
- `C[row][col]` → `C[row * N + col]`

CUDA kernels always operate on linear memory.

---

## Correct Mental Model

- One thread computes one output element
- Grid dimensions match the output matrix
- Index math defines correctness
- Parallelism comes from threads, not loops

---

## What This Does Not Cover

This document focuses on **correctness and understanding**.

It does not cover:
- Shared memory tiling
- Memory coalescing
- Tensor cores
- Performance optimization

Those are built on top of this foundation.
