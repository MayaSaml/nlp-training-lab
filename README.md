# NLP Training Lab (PyTorch)

Welcome! This repository is part of an educational series where I explore, document, and teach core NLP concepts using PyTorch — from foundational ideas like RNNs to more advanced topics like Transformers and LLM finetuning.

Each notebook is written with clarity and structure, meant to balance both practical understanding and professional code style. While some examples are minimal or toy-sized, the focus is on understanding *why* and *how* things work — a crucial step for mastering real-world NLP.

---

## Notebook Index

| Notebook | Description |
|----------|-------------|
| [`rnn_toy_examples.ipynb`](./rnn_toy_examples.ipynb) | Toy RNN-based models for 3 NLP tasks (language modeling, tagging, classification) |

More coming soon: Transformers, Attention, LLM Finetuning.

---

## File: `rnn_toy_examples.ipynb`

This notebook demonstrates how to build and train simple RNN-based models for three common NLP tasks:

### 1️. Language Modeling
> Predict the next word in a sequence based on previous context.

- Input: `so ➝ long ➝ and`
- Target: `long ➝ and ➝ thanks`
- Architecture: `Embedding ➝ RNN ➝ Linear ➝ Softmax`
- Output: Prediction of next word in sequence

### 2. POS Tagging
> Label each word in a sentence with its part-of-speech tag.

- Input: `Janet will back the bill`
- Target: `NNP MD VB DT NN`
- Architecture: `Embedding ➝ RNN ➝ Linear (tagset size)`
- Output: One tag per word (sequence-to-sequence)

### 3️. Text Classification
> Classify the sentiment of a sentence (positive/negative).

- Input: `so great` → positive, `bad bill` → negative
- Target: Binary labels
- Architecture: `Embedding ➝ RNN ➝ Last Hidden ➝ Linear`
- Output: Predicted class label

---

## How to Use
You can run the `.ipynb` notebook in:
- Jupyter Notebook or JupyterLab
- Google Colab

No special data or GPU is required — all examples are self-contained and CPU-friendly.

---

## About This Project
This repo reflects my teaching journey in NLP. It’s here to:
- Help others build foundational intuition
- Practice clear and effective PyTorch implementations
- Serve as a stepping stone to more complex topics
