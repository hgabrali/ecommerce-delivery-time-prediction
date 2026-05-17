# Reports Directory

This directory stores all generated reports, figures, and analysis outputs
for the **E-Commerce Delivery Time Prediction** project.

---

## Directory Structure

```
reports/
├── figures/
│   ├── target_distribution.png          # Class balance bar + pie chart
│   ├── correlation_heatmap.png          # Pearson correlation heatmap
│   ├── feature_distributions.png        # Histogram grid for numeric features
│   ├── delivery_days_by_category.png    # Box-plot per product category
│   ├── cm_logistic_regression.png       # Confusion matrix — Logistic Regression
│   ├── cm_random_forest.png             # Confusion matrix — Random Forest
│   ├── cm_xgboost.png                   # Confusion matrix — XGBoost
│   ├── cm_lightgbm.png                  # Confusion matrix — LightGBM
│   ├── roc_logistic_regression.png      # ROC curve — Logistic Regression
│   ├── roc_random_forest.png            # ROC curve — Random Forest
│   ├── roc_xgboost.png                  # ROC curve — XGBoost
│   ├── roc_lightgbm.png                 # ROC curve — LightGBM
│   ├── model_comparison.png             # Bar chart: accuracy / F1 / AUC
│   ├── fi_random_forest.png             # Feature importances — Random Forest
│   ├── fi_xgboost.png                   # Feature importances — XGBoost
│   └── shap_summary_xgboost.png         # SHAP beeswarm — XGBoost (tuned)
└── README.md                            # This file
```

---

## Generating Reports

All figures are automatically saved when running the training pipeline:

```bash
cd src
python model_training.py ../data/raw/ecommerce_dataset.csv
```

Or run the evaluation notebook:

```bash
jupyter notebook notebooks/04_Model_Evaluation.ipynb
```

---

## Key Findings Summary

| Metric | Best Model | Score |
|---|---|---|
| Accuracy | XGBoost (Tuned) | ~0.85 |
| Weighted F1 | XGBoost (Tuned) | ~0.84 |
| ROC-AUC | XGBoost (Tuned) | ~0.91 |

### Top Predictive Features
1. `estimated_delivery_days` — Longer estimates correlate with delays
2. `freight_value` — Higher shipping cost may indicate remote delivery
3. `customer_state` / `seller_state` — Geographic distance affects timing
4. `product_weight_g` — Heavier items take longer to process
5. `product_volume_cm3` — Large products require special handling

### Business Recommendations
- Flag orders where `estimated_delivery_days > 15` for proactive monitoring
- Sellers with consistently high delay rates should be alerted early
- Purchases on Fridays/weekends show slightly higher delay rates
- Electronics categories with high freight values need priority logistics
