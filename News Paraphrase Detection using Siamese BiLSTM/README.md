# News Paraphrase Detection using Siamese BiLSTM

Deep learning NLP project for detecting whether two news sentences convey the same meaning using a Siamese BiLSTM with Attention Pooling and Contrastive Loss.

## Dataset

- MRPC Dataset
- Total Pairs: 5,801
- Paraphrase: 3,900
- Non-Paraphrase: 1,901
- Train: 4,060
- Validation: 870
- Test: 871
- Vocabulary: 10,006
- Maximum Length: 40

## Model

Siamese BiLSTM with:
- 200-dimensional word embeddings
- 2-layer Bidirectional LSTM
- Attention Pooling
- LayerNorm
- GELU
- Dropout
- Contrastive Loss
- Distance-based classification
- Validation threshold optimization
- Early stopping

## Results

| Metric | Score |
|---|---:|
| Validation F1 | 57.64% |
| Test Accuracy | 61.54% |
| Test F1 | 51.38% |
| Best Threshold | 0.20 |

## Hardware

- Apple Silicon GPU
- PyTorch MPS acceleration

## Visualizations

- Training Loss
- Validation F1
- F1 vs Distance Threshold
- Pair Distance Distribution
- Confusion Matrix
- Test Performance

## Technologies

Python, PyTorch, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn

## Model File

siamese_bilstm_mrpc.pt

## Project Structure

- mrpc_dataset.csv
- siamese_bilstm_mrpc.pt
- siamese_bilstm.ipynb
- requirements.txt
- README.md

## Future Improvements

- BERT/RoBERTa embeddings
- Transformer-based Siamese architecture
- Hard-negative mining
- Triplet Loss
- Larger paraphrase datasets
- Semantic search integration

## Author

Ebbeju Lankapalli