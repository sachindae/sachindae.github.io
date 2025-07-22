---
layout: archive
title: "Optimizing Matrix Multiplication on CPU"
permalink: /blog/matmul
author_profile: true
---

I recently moved to a new job where the goal is to optimize LLM inference on specialized hardware. Since I've only been working on high-level frameworks (e.g. PyTorch and PySpark) the past two years, I've been brushing up on the lower-level details of computation. In this post, I'll start with a naive implementation of matrix multiplication in C that runs on CPU, and explore several optimizations.





