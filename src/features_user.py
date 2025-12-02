import pandas as pd
import numpy as np


def build_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit des features au niveau utilisateur à partir des transactions.

    Expects au minimum les colonnes :
    - user_id
    - product_amount
    - transaction_fee
    - cashback
    - loyalty_points
    - product_category
    - payment_method
    - device_type
    - location

    Returns
    -------
    pd.DataFrame
        DataFrame indexée par user_id avec des colonnes numériques prêtes pour modèle.
    """

    df = df.copy()

    if "transaction_date" in df.columns:
        df["transaction_datetime"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    else:
        df["transaction_datetime"] = pd.NaT

    numeric_agg = df.groupby("user_id").agg(
        nb_transactions=("transaction_id", "count") if "transaction_id" in df.columns else ("user_id", "count"),
        total_amount=("product_amount", "sum"),
        avg_amount=("product_amount", "mean"),
        std_amount=("product_amount", "std"),
        total_fee=("transaction_fee", "sum") if "transaction_fee" in df.columns else ("product_amount", lambda x: 0.0),
        total_cashback=("cashback", "sum") if "cashback" in df.columns else ("product_amount", lambda x: 0.0),
        total_loyalty_points=("loyalty_points", "sum") if "loyalty_points" in df.columns else ("product_amount", lambda x: 0.0),
    )

    numeric_agg["avg_fee"] = numeric_agg["total_fee"] / numeric_agg["nb_transactions"]
    numeric_agg["avg_cashback"] = numeric_agg["total_cashback"] / numeric_agg["nb_transactions"]
    numeric_agg["avg_loyalty_points"] = numeric_agg["total_loyalty_points"] / numeric_agg["nb_transactions"]

    numeric_agg["cashback_ratio"] = np.where(
        numeric_agg["total_amount"] > 0,
        numeric_agg["total_cashback"] / numeric_agg["total_amount"],
        0.0,
    )

    if "product_category" in df.columns:
        cat_amount = df.pivot_table(
            index="user_id",
            columns="product_category",
            values="product_amount",
            aggfunc="sum",
            fill_value=0.0,
        )
        cat_amount = cat_amount.div(cat_amount.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    else:
        cat_amount = pd.DataFrame(index=numeric_agg.index)

    if "payment_method" in df.columns:
        pay_counts = pd.crosstab(df["user_id"], df["payment_method"])
        pay_counts = pay_counts.div(pay_counts.sum(axis=1), axis=0)
    else:
        pay_counts = pd.DataFrame(index=numeric_agg.index)

    device_props = pd.DataFrame(index=numeric_agg.index)
    if "device_type" in df.columns:
        device_counts = pd.crosstab(df["user_id"], df["device_type"])
        device_props = device_counts.div(device_counts.sum(axis=1), axis=0)

    location_props = pd.DataFrame(index=numeric_agg.index)
    if "location" in df.columns:
        loc_counts = pd.crosstab(df["user_id"], df["location"])
        location_props = loc_counts.div(loc_counts.sum(axis=1), axis=0)

    features = numeric_agg.join([cat_amount, pay_counts, device_props, location_props], how="left")
    features = features.fillna(0.0)

    return features
