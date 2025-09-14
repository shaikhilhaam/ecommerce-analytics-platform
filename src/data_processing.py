# src/data_processing.py
import pandas as pd
from sqlalchemy.engine import Engine

def create_master_table_from_db(engine: Engine) -> pd.DataFrame:
    """
    Connects to the PostgreSQL database, runs a comprehensive SQL query to join all tables,
    and returns a single, clean master DataFrame.
    """
    
    query = """
    SELECT
        o.order_id,
        o.order_status,
        o.order_purchase_timestamp,
        o.order_approved_at,
        o.order_delivered_carrier_date,
        o.order_delivered_customer_date,
        o.order_estimated_delivery_date,
        c.customer_unique_id,
        c.customer_city,
        c.customer_state,
        c.customer_zip_code_prefix,
        oi.order_item_id,
        oi.product_id,
        oi.seller_id,
        oi.shipping_limit_date,
        oi.price,
        oi.freight_value,
        p.product_category_name_english AS product_category,
        prod.product_name_lenght,
        prod.product_description_lenght, 
        prod.product_photos_qty, 
        s.seller_city,
        s.seller_state,
        s.seller_zip_code_prefix,
        op.payment_sequential,
        op.payment_type,
        op.payment_installments,
        op.payment_value,
        rev.review_score,
        rev.review_comment_message
    FROM
        olist_orders o
    LEFT JOIN olist_customers c ON o.customer_id = c.customer_id
    LEFT JOIN olist_order_items oi ON o.order_id = oi.order_id
    LEFT JOIN olist_products prod ON oi.product_id = prod.product_id
    LEFT JOIN product_category_name_translation p ON prod.product_category_name = p.product_category_name
    LEFT JOIN olist_sellers s ON oi.seller_id = s.seller_id
    LEFT JOIN olist_order_payments op ON o.order_id = op.order_id
    LEFT JOIN olist_order_reviews rev ON o.order_id = rev.order_id;
    """
    
    print("Executing master SQL query to join all tables...")
    master_df = pd.read_sql(query, engine)
    print("Successfully loaded and merged data from PostgreSQL.")
    
    date_columns = [
        'order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
        'order_delivered_customer_date', 'order_estimated_delivery_date', 'shipping_limit_date'
    ]
    for col in date_columns:
        master_df[col] = pd.to_datetime(master_df[col], errors='coerce')
        
    return master_df