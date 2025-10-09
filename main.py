# main.py
import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from src.data_processing import create_master_table_from_db
from src.feature_engineering import create_propensity_features

def main():
    """Main function to run the full data processing and feature engineering pipeline."""
    load_dotenv()
    user, password = os.getenv('DB_USER'), os.getenv('DB_PASSWORD')
    host, port, db_name = os.getenv('DB_HOST'), os.getenv('DB_PORT'), os.getenv('DB_NAME')
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
    engine = create_engine(connection_string)

    # Step 1: Create master table from the database
    master_df = create_master_table_from_db(engine)

    # Step 2: Load the auxiliary geolocation dataset
    geo_df = pd.read_csv('data/raw/olist_geolocation_dataset.csv')

    # Step 3: Create the final features for the propensity model
    modeling_df = create_propensity_features(master_df, geo_df)

    # Step 4: Save the final modeling dataset
    modeling_data_path = 'data/processed/propensity_dataset_final.csv'
    modeling_df.to_csv(modeling_data_path, index=False)
    print(f"Final propensity modeling dataset saved to {modeling_data_path}")

if __name__ == "__main__":
    main()
