# Transformers

This folder contains clean and minimal PyTorch notebooks to help understand how Transformers work, step by step.

Each notebook focuses on one core idea — with real shapes, clear logic, and useful visualizations.

---

## Notebook Overview

### `01_self_attention_no_pe.ipynb`  
Simulates multi-head self-attention using real Transformer dimensions (`d_model = 512`, `num_heads = 8`).  
You’ll see how Q, K, V are created, how attention scores are computed, and how outputs from all heads are combined.

> No positional encoding yet — just pure attention logic.

---

### `02_tensor_indexing_and_pe.ipynb`  
Covers essential PyTorch tricks (indexing, broadcasting, squeeze/unsqueeze).  
Then explains how **sin/cos positional encoding** works with full math and clear plots.

> Helpful for understanding how input embeddings are enriched before attention.

---

### `03_self_attention_with_pe.ipynb`
Same as the first notebook, but this time with **positional encoding added** — just like in the real Transformer encoder.

---

## Reference

- [The Illustrated Transformer – Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
