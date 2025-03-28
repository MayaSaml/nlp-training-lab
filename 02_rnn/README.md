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
