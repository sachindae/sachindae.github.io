---
layout: archive
title: "Optimizing Matrix Multiplication on CPU"
permalink: /blog/matmul
author_profile: true
---

*Date: WIP*

In this post, we dive into matrix multiplication, the primary computation we execute to run an LLM. We'll start with a naive implementation of matrix multiplication in C that runs on CPU, and then explore several ways to improve the performance.

### Matrix Multiplication

As a refresher, let's first look at what matrix multiplication is doing. In the diagram below, we can see that to multiply two matrices, we take the dot product of each row of the first matrix with each row of the second matrix.

*Insert Image*

### Measuring Performance

In real ML models, it is common to multiply two really large matrices. For the purpose of measuring performance, we'll use square matrices of size 4096x4096.

## Experiments

Below, we implement matrix multiplication in several different ways to improve the performance.

### Hardware

Before getting started, I'm using a laptop with the following specs:
```
13th Gen Intel I5-1335U
- 8 Efficient Cores (3.4 GHz Max Turboboost)
- 2 Performance Cores (4.6 GHz Max Turboboost)
- 16 GB RAM (3200 MT/s)
```

### Naive MM

A naive implementation of matrix multiplication looks something like the following:

```
#define MATRIX_DIM 4096

int main() {
    float* A = malloc(MATRIX_DIM * MATRIX_DIM * sizeof(float));
    float* B = malloc(MATRIX_DIM * MATRIX_DIM * sizeof(float));
    float* C = malloc(MATRIX_DIM * MATRIX_DIM * sizeof(float));

    for(int i = 0; i < MATRIX_DIM; i++) {
        for(int j = 0; j < MATRIX_DIM; j++) {
            for(int k = 0; k < MATRIX_DIM; k++) {
                // C[i][j] = Dot Product(Row I of A, Column J of B)
                C[i*MATRIX_DIM + j] += A[i*MATRIX_DIM + k] * B[k*MATRIX_DIM + j];
            }
        }
    }
}
```

Looking at the naive matrix multiplication implementation from a compute and data access perspective:
- 2N<sup>3</sup> floating point operations (the 2 is from the add and multiply)
   - 137 GFLOPS
- 3N<sup>3</sup> 32-bit reads (read one value from A, B, and C per loop body iteration)
   - 824 GB reads
- N<sup>3</sup> 32-bit writes (write one value of C per loop body iteration)
   - 274 GB writes

Importantly, for every 2 FLOPS we need to move 16 bytes of data around. A term often used to represent this is
called arithmetic intensity, which is defined as the ratio of FLOPS to bytes. In this case, for every
2 FLOPS, we need to transfer 16 bytes to/from memory, so the arithmetic intensity is (1 / 8). However, a key
point to note here is that we don't necessarily transfer these bytes to/from memory in the implementation above.
Instead, we end up using the CPU's caches as a temporary faster storage than memory during the computation.

Given that the memory speed on my hardware is 3200 MT/s and each transfer is 8 bytes, this means I can transfer 
25.6 GB/s. Assuming I'm fully utilizing that memory transfer bandwidth, I'd only be able to do 3.2 GFLOPS of compute
which is far below the estimated HW theoretical max of 480 GFLOPS (10 cores * 3 GHz * 16 FLOPS). However, it's worth noting 

From timing the matrix multiplication above, it took ~1500 seconds, which puts the implementation at 0.09 GFLOPS.
The reason why it is so much lower than the estimated peak of 3.2 GFLOPS is due to it being challenging for a single CPU core to saturate memory bandwidth, especially when doing other operations at the same time.

### Re-Ordered Loop MM

Given that we'd expect even the naive implementation above which uses a single with non-vectorized instructions to get around 3 FLOPS (3 GHz single core), the memory accesses are definitely the bottleneck in the naive implementation.
One way we can improve memory access speed is to ensure we are reading contiguous data. In modern CPU architecture,
the cache line is typically 64 bytes. Concretely, when the CPU is loading data from memory, it will request chunks of 64 bytes. This means that contiguous memory accesses are more efficient than non-contiguous accesses to memory.

In the innermost loop of our naive implementation, we are accessing matrix B in a non contiguous way (we index into it using k*MATRIX_DIM, and the innermost loop is over the different values of k). A simple way to access B in a contiguous manner is to just swap the loop nesting order from `ijk` to `ikj`, as shown below:

```
for(int i = 0; i < MATRIX_DIM; i++) {
        for(int k = 0; k < MATRIX_DIM; k++) {
            for(int j = 0; j < MATRIX_DIM; j++) {
                // C[i][j] = Dot Product(Row I of A, Column J of B)
                C[i*MATRIX_DIM + j] += A[i*MATRIX_DIM + k] * B[k*MATRIX_DIM + j];
            }
        }
    }
```

Sure enough, this reduced the number of cache misses (e.g. memory accesses) significantly. Running cachegrind with a 512x512 matrix size (TODO: run on full size to get better memory access data):
- Naive Implementation: 134 million L1d cache misses, 50k L3 cache misses (e.g. memory accesses)
- Reordered Implementation: 8.5 million L1d cache misses, 50k L3 cache misses (e.g. memory accesses)

This memory access reduction lowered the overall runtime from 1500 seconds -> 250 seconds, a 6x speedup. 

### Tiled MM 



### References

Below are some great resources that I referred to before writing this post:
1. TODO
1. TODO





