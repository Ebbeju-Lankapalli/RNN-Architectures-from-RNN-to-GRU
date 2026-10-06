# Named Entity Recognition using BiLSTM

## Overview

This project implements a **Bidirectional LSTM (BiLSTM)** model for Named Entity Recognition (NER) on news articles.

The model identifies four entity types:
- Person (PER)
- Organization (ORG)
- Location (LOC)
- Miscellaneous (MISC)

## Dataset

The dataset contains news articles with BIO-formatted NER tags.

- ~17,000 sentences
- 9 BIO tags
- UTF-8 encoding

## Approach

1. Load and preprocess the NER dataset
2. Group words and tags by sentence
3. Build word and tag vocabularies
4. Convert words and tags into numerical sequences
5. Split data into training and validation sets
6. Pad sequences using PyTorch
7. Train a Bidirectional LSTM model
8. Evaluate using the seqeval F1 score
9. Visualize training performance and entity distributions

## Model

- Embedding Layer
- 2-layer Bidirectional LSTM
- 128-dimensional embeddings
- 128 hidden units
- Dropout: 0.3
- Linear classification layer
- 9 output BIO tags

## Results

| Metric | Result |
|---|---:|
| Validation F1 Score | **0.8528** |
| Vocabulary Size | **10,929** |
| Number of BIO Tags | **9** |

### Entity-Type Accuracy

| Entity | Accuracy |
|---|---:|
| PER | 88.31% |
| ORG | 80.60% |
| LOC | 89.32% |
| MISC | 78.60% |

The model achieved an F1 score of **0.8528**, exceeding the required minimum F1 score of **0.40**.

## Visualizations

The project includes:
- Training Loss
- Validation F1 Score
- Training Loss vs Validation F1
- Named Entity Distribution
- BIO Tag Distribution
- Entity-Type Prediction Accuracy
- Sample NER Predictions

## Tech Stack

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Seqeval
- Matplotlib

## Conclusion

The BiLSTM model successfully learned contextual and sequential patterns from news articles and accurately identified named entities across Person, Organization, Location, and Miscellaneous categories.