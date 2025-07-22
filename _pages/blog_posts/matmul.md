---
layout: archive
title: "Optimizing Matrix Multiplication on CPU"
permalink: /blog/matmul
author_profile: true
---

*Date: July 28, 2025*

I recently started a new job where we focus on optimizing LLM inference for specialized hardware. Since I only built things using high-level frameworks (e.g. PyTorch and PySpark) the past two years, I've been brushing up on the lower-level details of compute. In this post, we dive into matrix multiplication, the primary computation we execute to run an LLM. We'll start with a naive implementation of matrix multiplication in C that runs on CPU, and explore several ways to improve the performance.

### Matrix Multiplication

As a refresher, let's first look at what matrix multiplication is doing. In the diagram below, we can see that to multiply two matrices, we take the dot product of each row of the first matrix with each row of the second matrix.

*Insert Image*

### Measuring Performance

In real ML models, it is common to multiply two really large matrices. For the purpose of measuring performance, we'll use square matrices of size 4096x4096.

## Experiments

Below, we implement matrix multiplication in several different ways to improve the performance.

### Naive MM

```
#import <stdio.h>

#define MATRIX_DIM 4096

int main() {
    *float A = memalloc(MATRIX_DIM * MATRIX_DIM * sizeof(float));
    *float B = memalloc(MATRIX_DIM * MATRIX_DIM * sizeof(float));
    *float C = memalloc(MATRIX_DIM * MATRIX_DIM * sizeof(float));

    return 0;
}
```

### Memory Layout-Aware MM

### Matrix MM 

### Tiled MM 

### References

Below are some great resources that I referred to before writing this post:
1. TODO
1. TODO





