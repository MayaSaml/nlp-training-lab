# Transformers

This folder contains clean, minimal, and well-explained PyTorch notebooks that build up a **Transformer model step-by-step**.

Each notebook focuses on **one core idea**, with:
- Real tensor shapes
- Clear logic & modular code
- Helpful visualizations
- Educational comments throughout

---

## Notebook Overview

### 01_tensor_indexing_and_pe.ipynb
Fundamentals of PyTorch tensor slicing, indexing, and broadcasting.  
Also covers the **math and intuition** behind sinusoidal positional encodings — with plots.

### 02_self_attention_with_pe.ipynb
A toy simulation of **multi-head self-attention** with positional encoding.  
Shows how queries, keys, values are formed and how attention weights work.

### 03_encoder_block.ipynb
Builds a full **Transformer encoder block**:  
Self-attention → Add & Norm → Feedforward → Add & Norm.

### 04_encoder_stack.ipynb
Stacks 8 encoder blocks into a full **Transformer Encoder**.  
Follows the original paper’s architecture and dimensions.

### 05_decoder_stack.ipynb
Builds a full **Transformer Decoder** with masked self-attention + encoder-decoder cross-attention.

### 06_transformer_full.ipynb
Combines everything into:
- `Transformer`: Full encoder-decoder model
- `TransformerWithEmbeddings`: Adds embeddings and vocab projection
---

## Reference

- [The Illustrated Transformer – Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
