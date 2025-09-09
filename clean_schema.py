from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

def main():
    """
    Connects to the database and applies schema cleaning and structuring operations.
    """
    # Load environment variables
    load_dotenv()
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    host = os.getenv('DB_HOST')
    port = os.getenv('DB_PORT')
    db_name = os.getenv('DB_NAME')

    # Create database connection
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
    engine = create_engine(connection_string)

    # List of SQL commands to execute
    commands = [
        # (The list of commands is the same as before)
        "ALTER TABLE olist_orders ALTER COLUMN order_purchase_timestamp TYPE TIMESTAMP USING order_purchase_timestamp::timestamp;",
        "ALTER TABLE olist_orders ALTER COLUMN order_approved_at TYPE TIMESTAMP USING order_approved_at::timestamp;",
        "ALTER TABLE olist_orders ALTER COLUMN order_delivered_carrier_date TYPE TIMESTAMP USING order_delivered_carrier_date::timestamp;",
        "ALTER TABLE olist_orders ALTER COLUMN order_delivered_customer_date TYPE TIMESTAMP USING order_delivered_customer_date::timestamp;",
        "ALTER TABLE olist_orders ALTER COLUMN order_estimated_delivery_date TYPE TIMESTAMP USING order_estimated_delivery_date::timestamp;",
        "ALTER TABLE olist_order_items ALTER COLUMN shipping_limit_date TYPE TIMESTAMP USING shipping_limit_date::timestamp;",
        "ALTER TABLE olist_customers ADD PRIMARY KEY (customer_id);",
        "ALTER TABLE olist_orders ADD PRIMARY KEY (order_id);",
        "ALTER TABLE olist_products ADD PRIMARY KEY (product_id);",
        "ALTER TABLE olist_sellers ADD PRIMARY KEY (seller_id);",
        "ALTER TABLE olist_orders ADD CONSTRAINT fk_orders_customers FOREIGN KEY (customer_id) REFERENCES olist_customers(customer_id);",
        "ALTER TABLE olist_order_items ADD CONSTRAINT fk_order_items_orders FOREIGN KEY (order_id) REFERENCES olist_orders(order_id);",
        "ALTER TABLE olist_order_items ADD CONSTRAINT fk_order_items_products FOREIGN KEY (product_id) REFERENCES olist_products(product_id);",
        "ALTER TABLE olist_order_items ADD CONSTRAINT fk_order_items_sellers FOREIGN KEY (seller_id) REFERENCES olist_sellers(seller_id);",
        "ALTER TABLE olist_order_payments ADD CONSTRAINT fk_order_payments_orders FOREIGN KEY (order_id) REFERENCES olist_orders(order_id);",
        "ALTER TABLE olist_order_reviews ADD CONSTRAINT fk_order_reviews_orders FOREIGN KEY (order_id) REFERENCES olist_orders(order_id);"
    ]

    # Execute each command within a transaction
    with engine.connect() as connection:
        with connection.begin() as transaction: # <-- Start a transaction
            for command in commands:
                try:
                    # Wrap the command in the text() construct
                    connection.execute(text(command)) # <-- This is the corrected line
                    print(f"Successfully executed: {command}")
                except Exception as e:
                    print(f"Error executing command: {command}\n{e}")

if __name__ == '__main__':
    main()