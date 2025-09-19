# src/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from haversine import haversine, Unit

def create_propensity_features(master_df: pd.DataFrame, geo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the final, most advanced feature set for the repeat purchase propensity model.
    """
    # --- 1. Engineer Delivery & Seller Features (as before) ---
    delivered_df = master_df[master_df['order_status'] == 'delivered'].copy()
    seller_df = delivered_df.groupby('seller_id').agg(
        seller_avg_review_score=('review_score', 'mean'),
        seller_num_orders=('order_id', 'nunique')
    ).reset_index()

    first_purchase_df = master_df.loc[master_df.groupby('customer_unique_id')['order_purchase_timestamp'].idxmin()]

    # --- 2. Create the Target Variable (as before) ---
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
    # Merge seller features
    feature_df = pd.merge(first_purchase_df, seller_df, on='seller_id', how='left')

    # A) Advanced Geospatial: Customer-Seller Distance
    # Average coordinates for each zip code prefix for efficiency
    geo_coords = geo_df.groupby('geolocation_zip_code_prefix').agg(
        lat=('geolocation_lat', 'mean'),
        lng=('geolocation_lng', 'mean')
    ).reset_index()
    
    # Merge customer and seller coordinates
    feature_df = pd.merge(feature_df, geo_coords, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
    feature_df.rename(columns={'lat': 'customer_lat', 'lng': 'customer_lng'}, inplace=True)
    feature_df = pd.merge(feature_df, geo_coords, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
    feature_df.rename(columns={'lat': 'seller_lat', 'lng': 'seller_lng'}, inplace=True)

    # Calculate distance, handling potential missing coordinates
    feature_df['customer_seller_distance'] = feature_df.apply(
        lambda row: haversine((row['customer_lat'], row['customer_lng']), (row['seller_lat'], row['seller_lng']))
        if pd.notna(row['customer_lat']) and pd.notna(row['seller_lat']) else np.nan,
        axis=1
    )

    # B) Advanced Temporal: Cyclical Features
    feature_df['month_sin'] = np.sin(2 * np.pi * feature_df['order_purchase_timestamp'].dt.month / 12)
    feature_df['month_cos'] = np.cos(2 * np.pi * feature_df['order_purchase_timestamp'].dt.month / 12)
    feature_df['dayofweek_sin'] = np.sin(2 * np.pi * feature_df['order_purchase_timestamp'].dt.dayofweek / 7)
    feature_df['dayofweek_cos'] = np.cos(2 * np.pi * feature_df['order_purchase_timestamp'].dt.dayofweek / 7)

    # C) State-of-the-Art NLP: Text Embeddings
    # Fill missing reviews before embedding
    feature_df['review_comment_message'] = feature_df['review_comment_message'].fillna("no review")
    
    # Load a pre-trained model (this will download it on first run)
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # Create embeddings (this can take a few minutes)
    print("Creating text embeddings for review comments...")
    embeddings = embedding_model.encode(feature_df['review_comment_message'].to_list(), show_progress_bar=True)
    
    # Use PCA to reduce embedding dimensionality to 5 features
    pca = PCA(n_components=5)
    embeddings_reduced = pca.fit_transform(embeddings)
    for i in range(embeddings_reduced.shape[1]):
        feature_df[f'nlp_feature_{i+1}'] = embeddings_reduced[:, i]
        
    # --- 4. Finalizing the Modeling Dataset ---
    final_features = [
        'customer_unique_id', 'payment_value', 'payment_installments', 'review_score',
        'freight_value', 'seller_avg_review_score', 'seller_num_orders',
        'customer_seller_distance', 'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos',
        'nlp_feature_1', 'nlp_feature_2', 'nlp_feature_3', 'nlp_feature_4', 'nlp_feature_5'
    ]
    modeling_df = pd.merge(feature_df[final_features], target_df, on='customer_unique_id')

    # Handle any remaining missing values
    for col in modeling_df.columns:
        if modeling_df[col].isnull().any():
            modeling_df[col] = modeling_df[col].fillna(modeling_df[col].median())

    print("Final advanced feature engineering complete.")
    return modeling_df