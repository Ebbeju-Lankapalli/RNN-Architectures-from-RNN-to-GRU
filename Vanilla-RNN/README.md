# RNN Architectures — Vanilla RNN

This repository explores recurrent neural network architectures, starting with a **Vanilla RNN implemented from scratch using PyTorch**.

## 📌 Current Implementation

### Character-Level Text Generation using Vanilla RNN

A character-level Vanilla RNN is trained on a short story corpus to learn character patterns and generate new text from a given starting phrase.

### Key Concepts

- Character-level text preprocessing
- Character-to-index encoding
- One-hot encoding
- Vanilla RNN forward propagation
- Manual parameter initialization
- Backpropagation Through Time (BPTT)
- Cross-Entropy Loss
- Manual SGD updates
- Gradient clipping
- Perplexity evaluation
- Temperature-based text generation
- `torch.multinomial` sampling

## 🧠 Architecture

```text
Character Input
      ↓
One-Hot Encoding
      ↓
Vanilla RNN
      ↓
Hidden State (128)
      ↓
Output Layer
      ↓
Character Probabilities
      ↓
Generated Text
```

## Trainable Parameters

W_xh → Input → Hidden
W_hh → Hidden → Hidden
b_h  → Hidden Bias
W_hq → Hidden → Output
b_q  → Output Bias

## 📊 Results

* Hidden Size: 128
* Sequence Length: 25
* Batch Size: 16
* Epochs: 1000
* Initial Loss: ~3.30
* Final Loss: ~0.49
* Vocabulary Size: 27
* Gradient Clipping: 1.0
* Text generation temperature: 0.8

## 🛠️ Technologies

* Python
* PyTorch
* NumPy
* Matplotlib
* Jupyter Notebook


