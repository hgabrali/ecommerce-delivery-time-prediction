# E-Commerce: Predicting Product Delivery Time for an Electronics Store

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red.svg)

## Overview

This project builds machine learning models to predict whether an electronics
e-commerce order will be **delivered on time or delayed**. Early and accurate
identification of at-risk orders allows logistics teams to intervene proactively,
improving customer satisfaction and reducing financial losses.

The dataset is based on the Brazilian Olist e-commerce public dataset and includes
order details, product attributes, customer/seller locations, payment information,
and review scores.

---

## Business Problem

Late deliveries are one of the top drivers of negative customer reviews and
return requests in e-commerce. By predicting delivery delays **before** they
occur, the store can:

- Proactively notify customers of expected delays
- Prioritize expedited shipping for at-risk orders
- Identify sellers or routes that are consistently underperforming

---

## Project Structure

```
ecommerce-delivery-time-prediction/
├── data/
│   ├── raw/                    # Original, unmodified datasets
│   ├── processed/              # Cleaned and feature-engineered data
│   └── README.md               # Data dictionary and sources
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_Model_Evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Loading, cleaning, feature engineering
│   ├── model_training.py       # Model definitions, training, evaluation
│   └── visualization.py        # Plotting utilities
├── models/                     # Saved model artifacts (.pkl)
├── reports/
│   └── figures/                # Generated plots and charts
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Dataset

The dataset combines multiple tables from the Olist Brazilian E-Commerce dataset
(available on Kaggle). Key fields include:

| Feature | Description |
|---|---|
| `order_purchase_timestamp` | UTC timestamp when the order was placed |
| `order_delivered_customer_date` | Actual delivery date to the customer |
| `order_estimated_delivery_date` | Original estimated delivery date |
| `product_category_name` | Electronics sub-category |
| `product_weight_g` | Product weight in grams |
| `freight_value` | Shipping cost in BRL |
| `price` | Product price in BRL |
| `customer_state` / `seller_state` | Brazilian state codes |
| `review_score` | Customer satisfaction score (1–5) |

### Target Variable

`delivery_time_class`:
- **0** — Delivered on time or early
- **1** — Delivered late (actual date > estimated date)

---

## Machine Learning Pipeline

### 1. Data Preprocessing (`src/data_preprocessing.py`)
- Load and merge raw CSV files
- Engineer temporal features (delivery days, purchase hour/weekday, etc.)
- Handle missing values with median/mode imputation
- Label-encode categorical features
- Standard-scale numeric features
- Stratified train / validation / test split (70 / 10 / 20)

### 2. Model Training (`src/model_training.py`)

| Model | Description |
|---|---|
| Logistic Regression | Linear baseline with balanced class weights |
| Random Forest | Ensemble of 300 decision trees |
| XGBoost | Gradient boosting with early stopping |
| LightGBM | Fast gradient boosting with leaf-wise growth |
| XGBoost (Tuned) | Bayesian hyperparameter search via Optuna (50 trials) |

### 3. Evaluation Metrics
- Accuracy
- Weighted F1 Score
- ROC-AUC
- Confusion Matrix (normalized)
- Precision-Recall Curve

### 4. Interpretability
- Feature importances (built-in for tree models)
- SHAP beeswarm summary plots

---

## Installation

```bash
# Clone the repository
git clone https://github.com/hgabrali/ecommerce-delivery-time-prediction.git
cd ecommerce-delivery-time-prediction

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the full preprocessing + training pipeline

```bash
cd src
python model_training.py ../data/raw/ecommerce_dataset.csv
```

### Launch Jupyter notebooks

```bash
jupyter notebook notebooks/
```

### Run only preprocessing

```python
from src.data_preprocessing import run_preprocessing_pipeline

X_train, X_val, X_test, y_train, y_val, y_test, artifacts = \
    run_preprocessing_pipeline("data/raw/ecommerce_dataset.csv")
```

---

## Results (Expected)

| Model | Accuracy | F1 (Weighted) | ROC-AUC |
|---|---|---|---|
| Logistic Regression | ~0.74 | ~0.73 | ~0.79 |
| Random Forest | ~0.81 | ~0.80 | ~0.87 |
| XGBoost | ~0.83 | ~0.82 | ~0.89 |
| LightGBM | ~0.83 | ~0.82 | ~0.89 |
| XGBoost (Tuned) | ~0.85 | ~0.84 | ~0.91 |

> Note: Exact values depend on the dataset split and random seeds.

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to
discuss what you would like to change.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file.

---

## Author

**hgabrali** — [GitHub Profile](https://github.com/hgabrali)
