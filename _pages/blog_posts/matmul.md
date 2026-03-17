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
- 13th Gen Intel I5-1335U
   - 8 Efficient Cores (3.4 GHz Max Turboboost)
   - 2 Performance Cores (4.6 GHz Max Turboboost)
   - Caches (sum of all):
      - L1d: 352 KiB (10 instances)
      - L1i: 576 KiB (10 instances)
      - L2: 6.5 MiB (4 instances)
      - L3: 12 MiB (1 instance)
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
2 FLOPS, we need to transfer 16 bytes to/from memory, so the arithmetic intensity is (1 / 8).

Given that the memory speed on my hardware is 3200 MT/s and each transfer is 8 bytes, this means I can transfer 
25.6 GB/s. Assuming I'm fully utilizing that memory transfer bandwidth, I'd only be able to do 3.2 GFLOPS of compute
which is far below the estimated HW theoretical max of 480 GFLOPS (10 cores * 3 GHz * 16 FLOPS). However, a key
point to note here is that we don't necessarily transfer all data to/from memory in the implementation above.
Instead, we end up using the CPU's caches as a temporary faster storage than memory during the computation in
many cases.

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

Sure enough, this reduced the number of cache misses (e.g. memory accesses) significantly. Running cachegrind with a 512x512 matrix size.
- Naive Implementation: 134 million L1d cache misses, 50k L3 cache misses (e.g. memory accesses)
- Reordered Implementation: 8.5 million L1d cache misses, 50k L3 cache misses (e.g. memory accesses)

I'm too lazy to run cachegrind for the full 4096x4096 matrix size as it'd be extremely slow (since it simulates the caches), but the key idea is that the cache misses are significantly reduced with this access pattern. We don't see any difference in L3 cache misses since the matrices are ~1 MB each which easily fits into the 12 MB L3 cache on my machine.

When measuring this, I saw a 6x speedup (1500 seconds -> 250 seconds) in the runtime, which is around 0.55 GFLOPS. 

### Compiler-Optimized MM

Modern compilers can do quite a bit of optimization on your code when you tell them to (through compiler optimization flags). I'm using `gcc` for these experiments and the default optimization flag is `O0` which disables optimizations. When I passed in the `O3` flag, I saw a surprising 12.5x speedup (250 seconds -> 20 seconds), putting us at 6.9 GFLOPS. Above we discussed how we'd expect our implementation to reach at most ~3 GFLOPS assuming no vectorized assembly instructions are being used, so the compiler must be doing some sort of vectorization. Let's take a look at the assembly that the compiler generated:

```
.L3:
        movups %xmm ...
        movups %xmm ...
        mulps %xmm ...
        mulps %xmm ...
        addps %xmm ...
        addps %xmm ...
        ...
```

In x86-64 assembly the `xmm` registers are 128 bits wide, and we can see the assembly doing a pretty tightly packed loop of vectorized multiplications and additions using these registers. Below is the assembly for the unoptimized assembly (uses gcc default flag `O0`):

```
.L6:
        mulss %xmm ...
        ...
        ...
        addss %xmm ...
```

Interestingly, vector registers are used, but the `addss` and `mulss` operations only actually do operations on the low single precision floating-point values. 

### Parallelized MM

With just two small changes (loop ordering and a compiler optimization flag), we've been able to speed up a matmul between two 4096x4096 from 1500 seconds to 20 seconds, a whopping 75x speedup. This took our matrix multiplication implementation from 0.09 GFLOPS to 6.9 GFLOPS. Now that we have an implementation running on a single core with reasonable performance, we can try parallelizing the matrix multiplication.

In C, an easy way to parallelize programs is using OpenMP, an API for shared memory multi-processing. Below we add the `#pragma omp parallel for` directive to parallelize the outer for loop across all cores (1 thread per core):

```
#include <omp.h>

#pragma omp parallel for
for(int i = 0; i < MATRIX_DIM; i++) {
    for(int k = 0; k < MATRIX_DIM; k++) {
        #pragma omp simd
        for(int j = 0; j < MATRIX_DIM; j++) {
            C[i*MATRIX_DIM + j] += A[i*MATRIX_DIM + k] * B[k*MATRIX_DIM + j];
        }
    }
}
```

Note that I also needed to add the `#pragma omp simd` directive as I noticed the vectorization got removed from
the assembly when compiling with OpenMP initially. Ideally we'd expect around a 10x speedup since there are 10 physical cores, but this change only improved the runtime from 20 seconds to 14 seconds, a 1.42x speedup. This is most likely due to the additional memory/cache pressure that comes from having 10 threads running in parallel with the current implementation.

### Tiled MM
