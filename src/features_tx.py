import pandas as pd
import numpy as np


def build_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit des features au niveau transaction à partir du DataFrame brut.

    Expects au minimum :
    - product_amount
    - transaction_fee
    - cashback
    - loyalty_points
    - transaction_date
    - product_category
    - payment_method
    - device_type
    - location
    """

    df = df.copy()

    if "transaction_date" in df.columns:
        df["transaction_datetime"] = pd.to_datetime(df["transaction_date"], errors="coerce")
        df["tx_hour"] = df["transaction_datetime"].dt.hour
        df["tx_dayofweek"] = df["transaction_datetime"].dt.dayofweek
        df["tx_is_weekend"] = df["tx_dayofweek"].isin([5, 6]).astype(int)
    else:
        df["tx_hour"] = np.nan
        df["tx_dayofweek"] = np.nan
        df["tx_is_weekend"] = np.nan

    numeric_cols = []
    for col in ["product_amount", "transaction_fee", "cashback", "loyalty_points", "tx_hour", "tx_dayofweek", "tx_is_weekend"]:
        if col in df.columns:
            numeric_cols.append(col)

    tx_num = df[numeric_cols]

    cat_cols = []
    for col in ["product_category", "payment_method", "device_type", "location"]:
        if col in df.columns:
            cat_cols.append(col)

    if cat_cols:
        tx_cat = pd.get_dummies(df[cat_cols], drop_first=True)
        features = pd.concat([tx_num, tx_cat], axis=1)
    else:
        features = tx_num

    features = features.fillna(0.0)

    return features
