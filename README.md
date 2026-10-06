# RNN Architectures: From RNN to GRU

A practical deep learning repository covering **Recurrent Neural Network (RNN) architectures and sequence modeling**, progressing from a basic Vanilla RNN to more advanced BiLSTM and Siamese BiLSTM architectures, and finally to LSTM/GRU time-series forecasting.

This repository contains **four end-to-end deep learning projects** demonstrating how recurrent architectures can be applied to text generation, sequence labeling, semantic similarity, and time-series forecasting.

---

## Projects

### 1. Character-Level Text Generation using Vanilla RNN

A character-level language modeling project that trains a **Vanilla RNN** to learn sequential patterns from text and generate new text character by character.

#### Key Concepts

- Character-level language modeling
- Sequence-to-sequence prediction
- One-hot / indexed character representation
- Vanilla RNN
- Hidden-state propagation
- Cross-entropy loss
- Text generation
- Perplexity

#### Model Configuration

| Component | Value |
|---|---:|
| Architecture | Vanilla RNN |
| Vocabulary Size | 27 |
| Sequence Length | 25 |
| Hidden Size | 128 |
| Batch Size | 16 |
| Training Epochs | 1000 |
| Total Parameters | 23,451 |
| Trainable Parameters | 23,451 |
| Final Training Loss | 0.4894 |
| Perplexity | 1.57 |
| Loss Reduction | 85.15% |

#### Visualization

The project includes a complete training and model-performance dashboard showing:

- Training loss across 1000 epochs
- Dataset and feature statistics
- Initial vs final loss
- Perplexity
- Loss reduction
- Model parameter statistics
- Vanilla RNN architecture details

![Vanilla RNN Visualization](Vanilla_RNN_Visualization.png)

---

### 2. Named Entity Recognition using BiLSTM

A sequence-labeling project that uses a **Bidirectional LSTM (BiLSTM)** to identify named entities in text.

The model processes each sentence in both forward and backward directions, allowing it to use contextual information from both sides of a token.

#### Key Concepts

- Natural Language Processing
- Sequence labeling
- Named Entity Recognition
- BIO tagging
- Word embeddings
- Bidirectional LSTM
- Token-level classification
- Validation F1 evaluation

#### Dataset & Model Statistics

| Component | Value |
|---|---:|
| Total Sentences | 17,291 |
| Training Sentences | 13,832 |
| Validation Sentences | 3,459 |
| Vocabulary Size | 10,929 |
| BIO Tags | 9 |
| Maximum Sentence Length | 113 |
| Average Sentence Length | 14.7 tokens |
| Embedding Dimension | 128 |
| Hidden Dimension | 128 |
| BiLSTM Layers | 2 |
| Total Parameters | 2,060,681 |
| Trainable Parameters | 2,060,681 |
| Training Epochs | 15 |

#### Entity-Level Results

| Entity Type | Token Accuracy |
|---|---:|
| PER | 88.9% |
| ORG | 83.3% |
| LOC | 91.1% |
| MISC | 79.5% |

#### Final Result

- **Best Validation F1:** 0.8565
- **Final Validation F1:** 0.8565

#### Visualization

The dashboard contains:

- Training loss
- Validation F1 score
- Dataset statistics
- Vocabulary and BIO-tag statistics
- Maximum sentence length
- Entity-type prediction accuracy
- Model parameter information

![BiLSTM NER Visualization](BiLSTM_NER_Visualization.png)

---

### 3. News Paraphrase Detection using Siamese BiLSTM

A semantic similarity project that uses a **Siamese BiLSTM architecture** to determine whether two news sentences express the same meaning.

The two text inputs are processed using a shared sequence encoder, and their learned representations are compared to perform paraphrase detection.

#### Key Concepts

- Sentence similarity
- Paraphrase detection
- Siamese neural networks
- Shared-weight encoders
- BiLSTM
- Attention mechanism
- Contrastive learning
- Semantic sentence representations
- Threshold-based classification

#### Dataset & Model Statistics

| Component | Value |
|---|---:|
| Total Pairs | 5,801 |
| Training Pairs | 4,060 |
| Validation Pairs | 870 |
| Test Pairs | 871 |
| Vocabulary Size | 10,006 |
| Maximum Sequence Length | 40 |
| Paraphrase Pairs | 3,900 |
| Non-Paraphrase Pairs | 1,901 |
| Embedding Dimension | 200 |
| BiLSTM Hidden Dimension | 128 |
| BiLSTM Layers | 2 |
| Projection Dimension | 128 |
| Total Parameters | 2,997,937 |
| Trainable Parameters | 2,997,937 |
| Training Epochs | 18 |

#### Evaluation

| Metric | Result |
|---|---:|
| Test Accuracy | 62.11% |
| Test F1 | 51.75% |
| Best Validation F1 | 57.47% |
| Best Threshold | 0.205 |

#### Architecture

```text
Sentence A ──┐
             │
             ▼
       Shared BiLSTM
             │
          Attention
             │
             ▼
       Sentence Vector
             │
             ├──── Similarity ────► Classification
             │
Sentence B ──┘
```

#### Visualization

The dashboard shows:

- Contrastive training loss
- Validation F1 across epochs
- Dataset split statistics
- Paraphrase vs non-paraphrase distribution
- Test accuracy
- Test F1
- Best validation F1
- Classification threshold
- Siamese BiLSTM parameter statistics

![Siamese BiLSTM Paraphrase Visualization](Siamese_BiLSTM_Paraphrase_Visualization.png)

---

### 4. Weather Temperature Forecasting using LSTM & GRU

A time-series forecasting project comparing **LSTM and GRU** networks for temperature prediction.

Historical weather observations are converted into sliding-window sequences and used to predict the next temperature value.

#### Key Concepts

- Time-series forecasting
- Sliding-window sequences
- LSTM
- GRU
- Sequential feature learning
- Regression
- MAE
- RMSE
- Model comparison

#### Dataset & Configuration

| Component | Value |
|---|---:|
| Training Rows | 35,008 |
| Test Rows | 8,753 |
| Input Features | 3 |
| Features | Temperature, Humidity, Pressure |
| Window Size | 24 hours |
| Batch Size | 128 |
| LSTM Epochs | 30 |
| Hidden Size | 64 |
| Recurrent Layers | 2 |

#### Model Comparison

| Metric | LSTM | GRU |
|---|---:|---:|
| Parameters | 51,009 | 38,273 |
| RMSE | 0.5947 °C | **0.5804 °C** |
| MAE | 0.4231 °C | **0.4142 °C** |

**Best Model: GRU**

The GRU achieved a lower RMSE and MAE while also using fewer trainable parameters than the LSTM.

#### Visualization

The dashboard includes:

- Actual temperature vs LSTM predictions
- Actual temperature vs GRU predictions
- Dataset and feature statistics
- LSTM vs GRU RMSE comparison
- LSTM vs GRU MAE comparison
- Parameter comparison
- Best-model identification

![Weather LSTM GRU Visualization](Weather_LSTM_GRU_Visualization.png)

---

# Architecture Progression

The four projects demonstrate a progression through important recurrent architectures:

```text
Vanilla RNN
    │
    ▼
BiLSTM
    │
    ▼
Siamese BiLSTM + Attention
    │
    ▼
LSTM vs GRU
```

The repository therefore moves from basic recurrent sequence processing to bidirectional contextual learning, semantic sentence comparison, and practical time-series forecasting.

---

# Project Comparison

| Project | Architecture | Domain | Main Task |
|---|---|---|---|
| Character-Level Text Generation | Vanilla RNN | NLP | Text Generation |
| Named Entity Recognition | BiLSTM | NLP | Sequence Labeling |
| News Paraphrase Detection | Siamese BiLSTM | NLP | Sentence Similarity |
| Weather Temperature Forecasting | LSTM & GRU | Time Series | Forecasting |

---

# Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- PyTorch
- Natural Language Processing
- Recurrent Neural Networks
- LSTM
- GRU
- BiLSTM
- Attention Mechanism
- Siamese Networks

---

# Repository Structure

```text
RNN-Architectures-from-RNN-to-GRU/
│
├── Vanilla-RNN/
│   └── Character-Level-Text-Generation.ipynb
│
├── BiLSTM-NER/
│   └── Named-Entity-Recognition-BiLSTM.ipynb
│
├── Siamese-BiLSTM/
│   └── News-Paraphrase-Detection.ipynb
│
├── LSTM-GRU-Forecasting/
│   └── Weather-Temperature-Forecasting.ipynb
│
├── Vanilla_RNN_Visualization.png
├── BiLSTM_NER_Visualization.png
├── Siamese_BiLSTM_Paraphrase_Visualization.png
├── Weather_LSTM_GRU_Visualization.png
│
└── README.md
```

> File and folder names can be adjusted to match the exact notebook names in the repository.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Ebbeju-Lankapalli/RNN-Architectures-from-RNN-to-GRU.git
```

Move into the project:

```bash
cd RNN-Architectures-from-RNN-to-GRU
```

Create a Conda environment:

```bash
conda create -n rnn-project python=3.11 -y
```

Activate it:

```bash
conda activate rnn-project
```

Install the required packages:

```bash
pip install numpy pandas matplotlib scikit-learn torch jupyter
```

Launch Jupyter:

```bash
jupyter notebook
```

---

# Learning Outcomes

By completing this repository, you will understand:

- How recurrent neural networks process sequential data
- How hidden states carry information across time steps
- How Vanilla RNNs can be used for character-level generation
- Why bidirectional recurrent networks are useful for NLP
- How BiLSTM can perform sequence labeling
- How Siamese networks learn representations for sentence pairs
- How attention can improve sentence representation
- How contrastive learning can be used for similarity tasks
- How LSTM and GRU architectures can be applied to time-series forecasting
- How to evaluate recurrent models using task-specific metrics
- How to compare model accuracy, error metrics, and parameter counts

---

# Model Evaluation Summary

### Vanilla RNN

**Final Loss:** 0.4894  
**Perplexity:** 1.57  
**Loss Reduction:** 85.15%

### BiLSTM NER

**Best Validation F1:** 85.65%

### Siamese BiLSTM

**Test Accuracy:** 62.11%  
**Test F1:** 51.75%

### Weather Forecasting

**LSTM RMSE:** 0.5947 °C  
**GRU RMSE:** 0.5804 °C  
**Best Model:** GRU

---

# Visualizations

Each project contains a consolidated visualization dashboard designed to present the most important experimental results in a single image.

### Vanilla RNN

![Vanilla RNN Dashboard](Vanilla_RNN_Visualization.png)

### BiLSTM NER

![BiLSTM NER Dashboard](BiLSTM_NER_Visualization.png)

### Siamese BiLSTM

![Siamese BiLSTM Dashboard](Siamese_BiLSTM_Paraphrase_Visualization.png)

### LSTM & GRU Forecasting

![LSTM GRU Dashboard](Weather_LSTM_GRU_Visualization.png)

---

# Conclusion

This repository provides a practical progression through major **RNN-based deep learning architectures**.

Starting with a simple **Vanilla RNN**, the projects progressively introduce **BiLSTM**, **Siamese BiLSTM with Attention**, and **LSTM/GRU architectures**.

The projects demonstrate how recurrent networks can solve different classes of sequential problems:

```text
Text Generation
       ↓
Named Entity Recognition
       ↓
Semantic Similarity
       ↓
Time-Series Forecasting
```

Together, these projects form a practical foundation for understanding recurrent neural networks and their applications in **Natural Language Processing and sequential data modeling**.

---

## Author

**Ebbeju Lankapalli**

B.Tech — Computer Science and Engineering (AIML)

GitHub: [Ebbeju-Lankapalli](https://github.com/Ebbeju-Lankapalli)

---

## Repository

[**RNN Architectures: From RNN to GRU**](https://github.com/Ebbeju-Lankapalli/RNN-Architectures-from-RNN-to-GRU)
