"""
data_preprocessing.py
----------------------
Module for loading, cleaning, and preprocessing the e-commerce delivery
time dataset for an electronics store.

Author: hgabrali
Date: 2026
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TARGET_COLUMN = "delivery_time_class"   # 0 = on-time, 1 = delayed
NUMERIC_FEATURES = [
      "product_weight_g",
      "freight_value",
      "price",
      "payment_value",
      "product_length_cm",
      "product_height_cm",
      "product_width_cm",
      "customer_zip_code_prefix",
      "seller_zip_code_prefix",
      "review_score",
]
CATEGORICAL_FEATURES = [
      "product_category_name",
      "payment_type",
      "order_status",
      "customer_state",
      "seller_state",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(filepath: str) -> pd.DataFrame:
      """Load the raw CSV dataset.

          Parameters
              ----------
                  filepath : str
                          Absolute or relative path to the CSV file.

                              Returns
                                  -------
                                      pd.DataFrame
                                              Raw dataframe.
                                                  """
      if not os.path.exists(filepath):
                raise FileNotFoundError(f"Dataset not found at: {filepath}")
            logger.info("Loading data from %s", filepath)
    df = pd.read_csv(filepath, parse_dates=["order_purchase_timestamp",
                                                                                         "order_delivered_customer_date",
                                                                                         "order_estimated_delivery_date"])
    logger.info("Dataset shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
      """Derive new features from existing columns.

          Parameters
              ----------
                  df : pd.DataFrame
                          Raw dataframe.

                              Returns
                                  -------
                                      pd.DataFrame
                                              Dataframe with additional engineered features.
                                                  """
    df = df.copy()

    # Actual delivery time in days
    df["actual_delivery_days"] = (
              df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days

    # Estimated delivery window
    df["estimated_delivery_days"] = (
              df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.days

    # Delivery delay (positive = late, negative = early)
    df["delivery_delay_days"] = df["actual_delivery_days"] - df["estimated_delivery_days"]

    # Binary target: 1 if delivered late, 0 if on time or early
    df[TARGET_COLUMN] = (df["delivery_delay_days"] > 0).astype(int)

    # Product volume (cm^3)
    df["product_volume_cm3"] = (
              df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
    )

    # Price per gram (density proxy)
    df["price_per_gram"] = df["price"] / (df["product_weight_g"] + 1e-9)

    # Order purchase hour and weekday
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek

    logger.info("Feature engineering complete. New shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
      """Remove duplicates, invalid rows, and handle missing values.

          Parameters
              ----------
                  df : pd.DataFrame
                          Raw or partially processed dataframe.

                              Returns
                                  -------
                                      pd.DataFrame
                                              Cleaned dataframe.
                                                  """
      initial_shape = df.shape
      df = df.drop_duplicates()

    # Drop rows where target cannot be computed
      df = df.dropna(subset=["order_delivered_customer_date",
                                                         "order_purchase_timestamp",
                                                         "order_estimated_delivery_date"])

    # Remove physically impossible delivery times
    df = df[df["actual_delivery_days"] > 0]
    df = df[df["actual_delivery_days"] < 365]

    # Impute remaining missing numeric values with median
    num_imputer = SimpleImputer(strategy="median")
    df[NUMERIC_FEATURES] = num_imputer.fit_transform(df[NUMERIC_FEATURES])

    # Impute categorical missing values with the most frequent value
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[CATEGORICAL_FEATURES] = cat_imputer.fit_transform(df[CATEGORICAL_FEATURES])

    logger.info("Cleaned data: %s → %s", initial_shape, df.shape)
    return df


# ---------------------------------------------------------------------------
# Encoding & scaling
# ---------------------------------------------------------------------------
def encode_and_scale(
      df: pd.DataFrame,
      scaler_type: str = "standard",
) -> tuple[pd.DataFrame, dict]:
      """Encode categorical features and scale numeric features.

          Parameters
              ----------
                  df : pd.DataFrame
                          Cleaned and engineered dataframe.
                              scaler_type : str
                                      "standard" for StandardScaler or "minmax" for MinMaxScaler.

                                          Returns
                                              -------
                                                  tuple[pd.DataFrame, dict]
                                                          Transformed dataframe and a dict of fitted encoders/scalers.
                                                              """
      df = df.copy()
      artifacts = {}

    # Label-encode categoricals
      for col in CATEGORICAL_FEATURES:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                artifacts[f"le_{col}"] = le

    # Scale numerics
    scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
    all_numeric = NUMERIC_FEATURES + [
              "product_volume_cm3", "price_per_gram",
              "purchase_hour", "purchase_dayofweek",
              "actual_delivery_days", "estimated_delivery_days",
    ]
    existing_numeric = [c for c in all_numeric if c in df.columns]
    df[existing_numeric] = scaler.fit_transform(df[existing_numeric])
    artifacts["scaler"] = scaler

    logger.info("Encoding and scaling complete.")
    return df, artifacts


# ---------------------------------------------------------------------------
# Train / validation / test split
# ---------------------------------------------------------------------------
def split_dataset(
      df: pd.DataFrame,
      target_col: str = TARGET_COLUMN,
      test_size: float = 0.2,
      val_size: float = 0.1,
) -> tuple:
      """Split the dataset into training, validation, and test sets.

          Parameters
              ----------
                  df : pd.DataFrame
                          Fully preprocessed dataframe.
                              target_col : str
                                      Name of the target column.
                                          test_size : float
                                                  Proportion reserved for testing.
                                                      val_size : float
                                                              Proportion (of training data) reserved for validation.

                                                                  Returns
                                                                      -------
                                                                          tuple
                                                                                  X_train, X_val, X_test, y_train, y_val, y_test
                                                                                      """
      feature_cols = [c for c in df.columns if c != target_col]
      X = df[feature_cols]
      y = df[target_col]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
              X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    adjusted_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
              X_train_val, y_train_val,
              test_size=adjusted_val, random_state=RANDOM_STATE, stratify=y_train_val
    )

    logger.info(
              "Split sizes — Train: %d, Val: %d, Test: %d",
              len(X_train), len(X_val), len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------
def run_preprocessing_pipeline(filepath: str) -> tuple:
      """Execute the full preprocessing pipeline end-to-end.

          Parameters
              ----------
                  filepath : str
                          Path to raw CSV file.

                              Returns
                                  -------
                                      tuple
                                              X_train, X_val, X_test, y_train, y_val, y_test, artifacts
                                                  """
      df = load_data(filepath)
      df = engineer_features(df)
      df = clean_data(df)
      df, artifacts = encode_and_scale(df)
      X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)
      return X_train, X_val, X_test, y_train, y_val, y_test, artifacts


if __name__ == "__main__":
      import sys
      path = sys.argv[1] if len(sys.argv) > 1 else "data/ecommerce_dataset.csv"
      X_train, X_val, X_test, y_train, y_val, y_test, _ = run_preprocessing_pipeline(path)
      print(f"Training set size : {X_train.shape}")
      print(f"Validation set size: {X_val.shape}")
      print(f"Test set size      : {X_test.shape}")
  
