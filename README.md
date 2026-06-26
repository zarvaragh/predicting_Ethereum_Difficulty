# Predicting Ethereum Block Difficulty

Predict Ethereum blockchain difficulty using **ANN** and **LSTM** deep learning models, with a **Flask REST API** backend and a **React** frontend for visualization.

## Architecture

```
predicting_Ethereum_Difficulty/
├── Models/                  # Python training scripts
│   ├── singleFeature_ANN.py    # ANN on difficulty only
│   ├── singleFeature_LSTM.py   # LSTM on difficulty only
│   ├── Ann_multipleFeatures.py # ANN with gas_limit, gas_used, size, tx_count
│   └── LSTM_multipleFeatures.py
├── Back_end/python/Flask/   # REST API server
│   └── app.py
└── Front_end/
    ├── HTML/                # Vanilla HTML/JS frontend
    └── React/               # React frontend
```

## Models

| Model | Input Features | Architecture |
|-------|---------------|--------------|
| Single-feature ANN | `difficulty` (t-1) | Dense(50, relu) → Dense(1) |
| Single-feature LSTM | `difficulty` (t-1) | LSTM(50)×2 → Dense(1) |
| Multi-feature ANN | gas_limit, gas_used, size, tx_count | Dense(50, relu) → Dense(1) |
| Multi-feature LSTM | gas_limit, gas_used, size, tx_count | LSTM(50)×4 → Dense(1) |

## Requirements

```bash
pip install tensorflow flask flask-cors numpy pandas scikit-learn
```

## Training

```bash
cd predicting_Ethereum_Difficulty/Models
python singleFeature_ANN.py   # saves model_ANN.keras
python singleFeature_LSTM.py  # saves model_LSTM.keras
```

Data: `day.csv` — Ethereum daily block data with columns: `timestamp`, `difficulty`, `gas_limit`, `gas_used`, `size`, `transaction_count`.

## Running the API

```bash
cd predicting_Ethereum_Difficulty/Back_end/python/Flask/ethereum_pred_server
python app.py
```

The server starts on `http://localhost:5000`.

### Endpoints

**GET/POST** `/predict`

```json
{
  "gas_limit": 8000000,
  "gas_used":  7500000,
  "size":      20000,
  "transaction_count": 150
}
```

Response:
```json
{
  "ann_prediction":  [3.14e+15],
  "lstm_prediction": [3.09e+15]
}
```

## Tech Stack

- **TensorFlow 2.x / Keras** — model training and inference
- **Flask + Flask-CORS** — REST API
- **React** — interactive frontend with Chart.js visualization
- **scikit-learn** — MinMaxScaler preprocessing