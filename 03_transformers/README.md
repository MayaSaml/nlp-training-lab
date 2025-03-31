# Transformers

This folder contains clean and minimal PyTorch notebooks to help understand how Transformers work, step by step.

Each notebook focuses on one core idea — with real shapes, clear logic, and useful visualizations.

---

## Notebook Overview

### 01_tensor_indexing_and_pe.ipynb  
Explains PyTorch slicing, indexing, broadcasting, and squeeze/unsqueeze.  
Includes math + plots for sinusoidal positional encodings.  
Foundation for understanding how input tokens get position-aware vectors.

### 02_self_attention_with_pe.ipynb  
Simulates multi-head self-attention with positional encoding.  
You’ll see how Q, K, V are created, attention weights calculated, and multiple heads combined.

### 03_encoder_block.ipynb  
Implements a full Transformer encoder block: self-attention → residual + norm → feedforward → residual + norm.  
Result: contextualized (3, 512) token representations.

### 04_encoder_stack.ipynb
Implements a full stack of **8 Transformer encoder blocks** using PyTorch.

### 05_decoder_stack.ipynb
Implements a full stack of **8 Transformer decoder blocks** using PyTorch.

### 06_transformer_full.ipynb
Creates classes Transformer and TransformerWithEmbeddings incorporating previously created classes PositionalEncoding, TransformerEncoder, TransformerDecoder.


---

## Reference

- [The Illustrated Transformer – Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
