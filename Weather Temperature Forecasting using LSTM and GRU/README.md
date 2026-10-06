# Weather Temperature Forecasting using LSTM and GRU

## Overview

This project uses deep learning to forecast the next hour's temperature using the previous 24 hours of weather data.

The models use:
- Temperature
- Humidity
- Atmospheric Pressure

Two recurrent neural network architectures are implemented:
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)

## Dataset

The dataset contains hourly weather observations from a weather monitoring station in Jena, Germany.

Features:
- `datetime` — Timestamp
- `temperature` — Air temperature in °C
- `humidity` — Relative humidity
- `pressure` — Atmospheric pressure in mbar

A 24-hour sliding window is used to predict the temperature for the following hour.

## Workflow

1. Load the training and test datasets
2. Remove the `datetime` column
3. Normalize features using `MinMaxScaler`
4. Fit the scaler only on training data
5. Create 24-hour sequences
6. Train an LSTM model
7. Train a GRU model
8. Generate test predictions
9. Convert predictions back to °C
10. Evaluate using RMSE and MAE
11. Visualize predictions and errors

## Models

### LSTM
- 2 LSTM layers
- 64 hidden units
- Dropout: 0.2
- Adam optimizer
- Learning rate: 0.001

### GRU
- 2 GRU layers
- 64 hidden units
- Dropout: 0.2
- Adam optimizer
- Learning rate: 0.001

## Results

| Model | RMSE (°C) | MAE (°C) |
|---|---:|---:|
| LSTM | 0.6010 | 0.4281 |
| GRU | 0.5688 | 0.4021 |

Both models achieved an RMSE well below the required threshold of **3.0°C**.

## Visualizations

The project includes visualizations for:

- Historical temperature patterns
- Actual vs LSTM predictions
- Actual vs GRU predictions
- LSTM vs GRU predictions
- Prediction errors
- Prediction error distributions
- Model RMSE and MAE comparison

## Tech Stack

- Python
- NumPy
- Pandas
- PyTorch
- Scikit-learn
- Matplotlib

## Hardware

Training was performed using Apple Silicon **MPS** acceleration on a MacBook Air M4 when available.

## Conclusion

Both LSTM and GRU successfully learned short-term temporal patterns in the weather data and produced accurate next-hour temperature forecasts. The GRU achieved a slightly lower RMSE and MAE than the LSTM in this experiment.

## Required Outputs

```python
lstm_predictions
gru_predictions