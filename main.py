# main.py
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
from src.data_processing import create_master_table_from_db
from src.feature_engineering import create_ltv_features

def main():
    """Main function to run the data processing and feature engineering pipeline."""
    load_dotenv()
    # (Database connection setup remains the same)
    user, password = os.getenv('DB_USER'), os.getenv('DB_PASSWORD')
    host, port, db_name = os.getenv('DB_HOST'), os.getenv('DB_PORT'), os.getenv('DB_NAME')
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
    engine = create_engine(connection_string)

    # Step 1: Create master table from the database
    master_df = create_master_table_from_db(engine)

    # Step 2: Create features for LTV modeling
    modeling_df = create_ltv_features(master_df)

    # Step 3: Save the final modeling dataset
    modeling_data_path = 'data/processed/modeling_dataset.csv'
    modeling_df.to_csv(modeling_data_path, index=False)
    print(f"Modeling dataset saved to {modeling_data_path}")

if __name__ == "__main__":
    main()