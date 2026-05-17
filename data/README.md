# Data Directory

This directory contains the raw and processed datasets used in the
**E-Commerce: Predicting Product Delivery Time for an Electronics Store** project.

---

## Directory Structure

```
data/
├── raw/                          # Original, unmodified source files
│   ├── olist_orders_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_order_payments_dataset.csv
│   ├── olist_order_reviews_dataset.csv
│   ├── olist_customers_dataset.csv
│   ├── olist_sellers_dataset.csv
│   ├── olist_products_dataset.csv
│   └── product_category_name_translation.csv
├── processed/
│   ├── ecommerce_merged.csv      # Merged and joined dataset
│   ├── ecommerce_features.csv    # After feature engineering
│   └── ecommerce_final.csv       # Cleaned, encoded, and scaled
└── README.md                     # This file
```

---

## Data Source

The dataset is derived from the **Brazilian E-Commerce Public Dataset by Olist**,
available on [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

> License: CC BY-NC-SA 4.0

---

## Data Dictionary

### Core Order Table (`olist_orders_dataset.csv`)

| Column | Type | Description |
|---|---|---|
| `order_id` | string | Unique order identifier |
| `customer_id` | string | Key to the customer dataset |
| `order_status` | string | Order status (delivered, shipped, etc.) |
| `order_purchase_timestamp` | datetime | Purchase timestamp (UTC) |
| `order_approved_at` | datetime | Payment approval timestamp |
| `order_delivered_carrier_date` | datetime | Date order was handed to logistics |
| `order_delivered_customer_date` | datetime | Actual delivery date to customer |
| `order_estimated_delivery_date` | datetime | Estimated delivery date at purchase time |

### Order Items (`olist_order_items_dataset.csv`)

| Column | Type | Description |
|---|---|---|
| `order_id` | string | Order reference |
| `order_item_id` | int | Sequential item number within order |
| `product_id` | string | Product reference |
| `seller_id` | string | Seller reference |
| `price` | float | Item price in BRL |
| `freight_value` | float | Shipping cost in BRL |

### Products (`olist_products_dataset.csv`)

| Column | Type | Description |
|---|---|---|
| `product_id` | string | Unique product identifier |
| `product_category_name` | string | Portuguese category name |
| `product_weight_g` | float | Weight in grams |
| `product_length_cm` | float | Length in cm |
| `product_height_cm` | float | Height in cm |
| `product_width_cm` | float | Width in cm |

### Customers (`olist_customers_dataset.csv`)

| Column | Type | Description |
|---|---|---|
| `customer_id` | string | Unique customer key |
| `customer_zip_code_prefix` | string | 5-digit ZIP prefix |
| `customer_city` | string | Customer city |
| `customer_state` | string | Brazilian state abbreviation |

### Sellers (`olist_sellers_dataset.csv`)

| Column | Type | Description |
|---|---|---|
| `seller_id` | string | Unique seller key |
| `seller_zip_code_prefix` | string | 5-digit ZIP prefix |
| `seller_city` | string | Seller city |
| `seller_state` | string | Brazilian state abbreviation |

### Payments (`olist_order_payments_dataset.csv`)

| Column | Type | Description |
|---|---|---|
| `order_id` | string | Order reference |
| `payment_sequential` | int | Payment installment number |
| `payment_type` | string | credit_card, boleto, voucher, debit_card |
| `payment_installments` | int | Number of installments |
| `payment_value` | float | Transaction value in BRL |

### Reviews (`olist_order_reviews_dataset.csv`)

| Column | Type | Description |
|---|---|---|
| `review_id` | string | Unique review identifier |
| `order_id` | string | Order reference |
| `review_score` | int | Star rating 1-5 |
| `review_creation_date` | datetime | When review was sent |
| `review_answer_timestamp` | datetime | When review was answered |

---

## Engineered Features

After running `src/data_preprocessing.py`, the following derived features are added:

| Feature | Description |
|---|---|
| `actual_delivery_days` | Days from purchase to actual delivery |
| `estimated_delivery_days` | Days from purchase to estimated delivery |
| `delivery_delay_days` | actual - estimated (positive = late) |
| `delivery_time_class` | **Target**: 0 = on-time, 1 = delayed |
| `product_volume_cm3` | length × height × width |
| `price_per_gram` | price / weight (density proxy) |
| `purchase_hour` | Hour of day when order was placed |
| `purchase_dayofweek` | Day of week (0=Monday … 6=Sunday) |

---

## Notes

- Only orders with `order_status == "delivered"` and non-null delivery dates are retained.
- Orders with `actual_delivery_days <= 0` or `> 365` are considered erroneous and removed.
- Category names are translated from Portuguese to English using the translation table.
- All numeric features are standardized (zero mean, unit variance) before model training.
