# src/feature_engineering.py
import pandas as pd
import numpy as np

def create_ltv_features(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the feature set and target variable for the LTV prediction model.
    """
    # Find the first purchase for each customer
    first_purchase_df = master_df.loc[master_df.groupby('customer_unique_id')['order_purchase_timestamp'].idxmin()]

    # --- Create the Target Variable (ltv_90_days) ---
    # FIX: Added include_groups=False to silence the FutureWarning
    target_df = master_df.groupby('customer_unique_id', group_keys=False).apply(
        lambda x: x[
            (x['order_purchase_timestamp'] > x['order_purchase_timestamp'].min()) &
            (x['order_purchase_timestamp'] <= x['order_purchase_timestamp'].min() + pd.Timedelta(days=90))
        ]['payment_value'].sum(),
        include_groups=False
    ).reset_index(name='ltv_90_days')

    # --- Create Features from the First Purchase ---
    feature_df = first_purchase_df[[
        'customer_unique_id', 'payment_value', 'payment_installments', 'review_score',
        'freight_value', 'product_category', 'product_photos_qty', 'product_description_lenght'
    ]].copy()

    # --- Combine into the Final Modeling Dataset ---
    modeling_df = pd.merge(feature_df, target_df, on='customer_unique_id')

    # FIX: Avoid inplace=True and use direct assignment to prevent SettingWithCopyWarning
    modeling_df['review_score'] = modeling_df['review_score'].fillna(modeling_df['review_score'].median())
    modeling_df['product_photos_qty'] = modeling_df['product_photos_qty'].fillna(modeling_df['product_photos_qty'].median())
    modeling_df['product_description_lenght'] = modeling_df['product_description_lenght'].fillna(modeling_df['product_description_lenght'].median())
    modeling_df['product_category'] = modeling_df['product_category'].fillna('unknown')

    # Convert categorical feature into numerical using one-hot encoding
    modeling_df = pd.get_dummies(modeling_df, columns=['product_category'], drop_first=True, dtype=int)

    print("Feature engineering for LTV prediction complete.")
    return modeling_df