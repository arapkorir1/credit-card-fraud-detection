# Dataset Card: Credit Card Fraud Detection

## Source
- Kaggle: [mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- License: Open Database License (ODbL) — see Kaggle page for full terms
- Original source: transactions by European cardholders, September 2013,
  provided by Worldline and the Machine Learning Group (ULB)

## File
- `data/raw/creditcard.csv`
- SHA-256: `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`
- Size: 150.83 MB

## Schema
| Column | Type | Description |
|---|---|---|
| `Time` | float | Seconds elapsed between this transaction and the first transaction in the dataset |
| `V1`-`V28` | float | PCA-transformed features (anonymized, no domain meaning available) |
| `Amount` | float | Transaction amount |
| `Class` | int (0/1) | Target: 1 = fraud, 0 = legitimate |

## Class Balance
| Class | Count | Percentage |
|---|---|---|
| Legitimate (0) | 284,315 | 99.827% |
| Fraud (1) | 492 | 0.173% |
| **Total** | **284,807** | **100%** |

⚠️ Severe class imbalance — accuracy is not a meaningful metric here.
See Phase 7/8 for how this is handled (PR-AUC, class weighting, SMOTE, threshold tuning).

## Known caveats
- `V1`-`V28` are PCA components of original features withheld for
  confidentiality — no domain-driven feature engineering is possible on them.
- `Time` is relative (seconds since first transaction), not a wall-clock timestamp.
- Data covers only 2 days of transactions — temporal generalization beyond
  this window is untested.
