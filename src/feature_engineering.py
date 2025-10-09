# src/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from haversine import haversine

def create_propensity_features(master_df: pd.DataFrame, geo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the final, most advanced feature set for the repeat purchase propensity model.
    """
    # --- 1. Engineer Delivery & Seller Features ---
    delivered_df = master_df[master_df['order_status'] == 'delivered'].copy()
    seller_df = delivered_df.groupby('seller_id').agg(
        seller_avg_review_score=('review_score', 'mean'),
        seller_num_orders=('order_id', 'nunique')
    ).reset_index()

    first_purchase_df = master_df.loc[master_df.groupby('customer_unique_id')['order_purchase_timestamp'].idxmin()]

    # --- 2. Create the Target Variable ---
    ltv_df = master_df.groupby('customer_unique_id', group_keys=False).apply(
        lambda x: x[
            (x['order_purchase_timestamp'] > x['order_purchase_timestamp'].min()) &
            (x['order_purchase_timestamp'] <= x['order_purchase_timestamp'].min() + pd.Timedelta(days=90))
        ]['payment_value'].sum(),
        include_groups=False
    ).reset_index(name='ltv_90_days')
    ltv_df['is_repeat'] = (ltv_df['ltv_90_days'] > 0).astype(int)
    target_df = ltv_df[['customer_unique_id', 'is_repeat']]
    
    # --- 3. Engineer ADVANCED Features ---
    feature_df = pd.merge(first_purchase_df, seller_df, on='seller_id', how='left')
    
    # A) Delivery Performance
    feature_df['delivery_time_vs_estimated'] = \
        (feature_df['order_estimated_delivery_date'] - feature_df['order_delivered_customer_date']).dt.days
    feature_df['first_order_was_early'] = \
        (feature_df['delivery_time_vs_estimated'] > 0).astype(int)

    # B) Geospatial: Customer-Seller Distance
    geo_coords = geo_df.groupby('geolocation_zip_code_prefix').agg(
        lat=('geolocation_lat', 'mean'),
        lng=('geolocation_lng', 'mean')
    ).reset_index()
    feature_df = pd.merge(feature_df, geo_coords, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
    feature_df.rename(columns={'lat': 'customer_lat', 'lng': 'customer_lng'}, inplace=True)
    feature_df = pd.merge(feature_df, geo_coords, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
    feature_df.rename(columns={'lat': 'seller_lat', 'lng': 'seller_lng'}, inplace=True)
    feature_df['customer_seller_distance'] = feature_df.apply(
        lambda row: haversine((row['customer_lat'], row['customer_lng']), (row['seller_lat'], row['seller_lng']))
        if pd.notna(row['customer_lat']) and pd.notna(row['seller_lat']) else np.nan,
        axis=1
    )

    # C) Temporal: Cyclical Features
    feature_df['month_sin'] = np.sin(2 * np.pi * feature_df['order_purchase_timestamp'].dt.month / 12)
    feature_df['month_cos'] = np.cos(2 * np.pi * feature_df['order_purchase_timestamp'].dt.month / 12)

    # D) NLP: Text Embeddings
    feature_df['review_comment_message'] = feature_df['review_comment_message'].fillna("no review")
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("Creating text embeddings for review comments... (this may take a few minutes)")
    embeddings = embedding_model.encode(feature_df['review_comment_message'].to_list(), show_progress_bar=True)
    pca = PCA(n_components=5)
    embeddings_reduced = pca.fit_transform(embeddings)
    for i in range(embeddings_reduced.shape[1]):
        feature_df[f'nlp_feature_{i+1}'] = embeddings_reduced[:, i]
        
    # --- 4. Finalizing the Modeling Dataset ---
    final_features = [
        'customer_unique_id', 'payment_value', 'payment_installments', 'review_score',
        'freight_value', 'seller_avg_review_score', 'seller_num_orders',
        'customer_seller_distance', 'month_sin', 'month_cos',
        'delivery_time_vs_estimated', 'first_order_was_early',
        'nlp_feature_1', 'nlp_feature_2', 'nlp_feature_3', 'nlp_feature_4', 'nlp_feature_5',
        'product_category', 'customer_state' # Keep categorical for one-hot encoding
    ]
    modeling_df = pd.merge(feature_df[final_features], target_df, on='customer_unique_id')

    # Handle any remaining missing values
    for col in modeling_df.select_dtypes(include=np.number).columns:
        if modeling_df[col].isnull().any():
            modeling_df[col] = modeling_df[col].fillna(modeling_df[col].median())
            
    modeling_df['product_category'] = modeling_df['product_category'].fillna('unknown')
    
    cat_cols = ['product_category', 'customer_state']
    modeling_df = pd.get_dummies(modeling_df, columns=cat_cols, drop_first=True, dtype=int)

    print("Final advanced feature engineering complete.")
    return modeling_df
