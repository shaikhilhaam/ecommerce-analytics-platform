# src/feature_engineering.py
import pandas as pd
import numpy as np

def create_propensity_features(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the final, most advanced feature set for the repeat purchase propensity model.
    """
    # --- Create Seller-level Features (as before) ---
    delivered_df = master_df[master_df['order_status'] == 'delivered'].copy()
    seller_df = delivered_df.groupby('seller_id').agg(
        seller_avg_review_score=('review_score', 'mean'),
        seller_num_orders=('order_id', 'nunique')
    ).reset_index()

    # Find the first purchase for each customer
    first_purchase_df = master_df.loc[master_df.groupby('customer_unique_id')['order_purchase_timestamp'].idxmin()]

    # --- Create the Target Variable (is_repeat) ---
    ltv_df = master_df.groupby('customer_unique_id', group_keys=False).apply(
        lambda x: x[
            (x['order_purchase_timestamp'] > x['order_purchase_timestamp'].min()) &
            (x['order_purchase_timestamp'] <= x['order_purchase_timestamp'].min() + pd.Timedelta(days=90))
        ]['payment_value'].sum(),
        include_groups=False
    ).reset_index(name='ltv_90_days')
    ltv_df['is_repeat'] = (ltv_df['ltv_90_days'] > 0).astype(int)
    target_df = ltv_df[['customer_unique_id', 'is_repeat']]

    # --- Engineer NEW Advanced Features ---
    # 1. Delivery Performance Features for the first order
    first_purchase_df['delivery_time_vs_estimated'] = \
        (first_purchase_df['order_estimated_delivery_date'] - first_purchase_df['order_delivered_customer_date']).dt.days
    
    # 2. Was the first order early?
    first_purchase_df['first_order_was_early'] = \
        (first_purchase_df['delivery_time_vs_estimated'] > 0).astype(int)

    # --- Create Features from the First Purchase ---
    feature_df = pd.merge(first_purchase_df, seller_df, on='seller_id', how='left')
    feature_df['first_purchase_month'] = feature_df['order_purchase_timestamp'].dt.month
    feature_df['first_purchase_dayofweek'] = feature_df['order_purchase_timestamp'].dt.dayofweek
    
    feature_df = feature_df[[
        'customer_unique_id', 'payment_value', 'payment_installments', 'review_score',
        'freight_value', 'product_category', 'customer_state',
        'first_purchase_month', 'first_purchase_dayofweek',
        'seller_avg_review_score', 'seller_num_orders',
        'delivery_time_vs_estimated', # <-- NEW advanced feature
        'first_order_was_early'       # <-- NEW advanced feature
    ]].copy()
    
    modeling_df = pd.merge(feature_df, target_df, on='customer_unique_id')

    # Handle missing values
    # delivery_time_vs_estimated might have NaNs if dates were missing
    modeling_df['delivery_time_vs_estimated'] = modeling_df['delivery_time_vs_estimated'].fillna(modeling_df['delivery_time_vs_estimated'].median())
    modeling_df['review_score'] = modeling_df['review_score'].fillna(modeling_df['review_score'].median())
    modeling_df['seller_avg_review_score'] = modeling_df['seller_avg_review_score'].fillna(modeling_df['seller_avg_review_score'].median())
    modeling_df['seller_num_orders'] = modeling_df['seller_num_orders'].fillna(0)
    modeling_df['product_category'] = modeling_df['product_category'].fillna('unknown')
    
    cat_cols = ['product_category', 'customer_state']
    modeling_df = pd.get_dummies(modeling_df, columns=cat_cols, drop_first=True, dtype=int)

    print("Final advanced feature engineering complete.")
    return modeling_df