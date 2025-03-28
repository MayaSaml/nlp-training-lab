# 🧠 Multi-Head Self-Attention (Toy Example)

This notebook demonstrates how **multi-head self-attention** works using a minimal PyTorch implementation — with real Transformer dimensions:

- `d_model = 512`
- `num_heads = 8`
- `d_k = 64`

We simulate attention over the sentence **"I understand this"** using randomly initialized token embeddings.

---

## 🔍 What Happens Here

- Project input embeddings into Q, K, V for each head
- Compute scaled dot-product attention per head
- Concatenate all 8 attention outputs (shape: `3 × 512`)
- Apply final linear projection (`W_O`)

This matches the output of the Multi-Head Attention block in the original Transformer encoder — ready to be passed to LayerNorm and the feedforward layer.

---

## 📎 Output

The final output is a matrix of shape `(3, 512)`, representing the attention-enhanced vector for each input token.

---

## 🛠 Notes

- Each head uses its own random `W_Q`, `W_K`, and `W_V` for clarity
- In real models, these would be trainable parameters shared across batches

## 📚 Reference

Inspired by the visual explanation in  
**[The Illustrated Transformer by Jay Alammar](https://jalammar.github.io/illustrated-transformer/)**


