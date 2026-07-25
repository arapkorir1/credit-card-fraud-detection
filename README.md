# Credit Card Fraud Detection

End-to-end machine learning project detecting fraudulent credit card transactions
on a severely imbalanced dataset (0.17% positive class), from raw data to a
deployed, containerized inference service.

## Problem

Given anonymized, PCA-transformed transaction features, predict whether a
transaction is fraudulent. The core challenge is extreme class imbalance
(492 fraud cases out of 284,807 transactions), which makes naive accuracy
metrics meaningless and requires careful choices around resampling, class
weighting, evaluation metric (PR-AUC over ROC-AUC), and decision threshold.

## Dataset

- Source: [Kaggle - Credit Card Fraud Detection (mlg-ulb/creditcardfraud)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions, 492 labeled fraud (0.172%)
- Features `V1`-`V28`: PCA-transformed (anonymized, no domain meaning)
- Features `Time`, `Amount`: raw
- Target: `Class` (0 = legitimate, 1 = fraud)

## Project Status

🚧 In progress — see phases below.

## Tech Stack

- **Modeling:** scikit-learn, CatBoost, XGBoost, LightGBM, imbalanced-learn, Optuna
- **Explainability:** SHAP
- **API:** FastAPI
- **UI:** Streamlit
- **Deployment:** Docker
- **Env management:** conda
- **Language:** Python 3.11

## Project Structure

\`\`\`
credit-card-fraud-detection/
├── data/               # raw/processed data (gitignored)
├── notebooks/          # EDA, feature engineering, modeling, evaluation
├── src/                # reusable pipeline code (data, features, models, evaluate, predict)
├── api/                # FastAPI inference service
├── app/                # Streamlit frontend
├── models/             # trained model artifacts (gitignored)
├── reports/            # figures, metrics, evaluation outputs
├── tests/              # unit tests
├── monitoring/         # drift/monitoring scripts
├── environment.yml
└── README.md
\`\`\`

## Setup

\`\`\`bash
conda env create -f environment.yml
conda activate fraud-detection
\`\`\`

## Results

_TBD — filled in after Phase 8 (Evaluation)._

## License

MIT
