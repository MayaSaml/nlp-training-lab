# Transformer Training Pipeline (PyTorch)

This folder contains the **training-ready code** for running Transformer models on real data.

While `03_transformers/` walks through individual Transformer components, here we move to:
- Clean reusable modules
- Tokenization & data prep
- Training & evaluation
- Inference & decoding

---

## Folder Contents

### transformer.py
Core implementation of the model, including:
- `PositionalEncoding`
- `TransformerEncoder`
- `TransformerDecoder`
- `Transformer`
- `TransformerWithEmbeddings`

Modularized and ready to plug into any NLP task (e.g., translation, summarization).

---

## Notebooks

### 01_prepare_data.ipynb 
Tokenization, vocab setup, and dataloaders.

### 02_train_loop.ipynb  
Training logic, optimizer, scheduler, loss, accuracy.

### 03_generate.ipynb  
How to decode output from the model autoregressively.

---

## Resources

- Based on the official Transformer paper:  
  [“Attention Is All You Need”](https://arxiv.org/abs/1706.03762)

- Visualizations inspired by:  
  [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

---

## Goals

- Keep everything **simple, clear, and readable**
- Avoid unnecessary abstraction
- Focus on **learning by building**
