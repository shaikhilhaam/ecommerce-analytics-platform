# main.py
import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from src.data_processing import create_master_table_from_db
from src.feature_engineering import create_propensity_features

def main():
    """Main function to run the data processing and feature engineering pipeline."""
    load_dotenv()
    user, password = os.getenv('DB_USER'), os.getenv('DB_PASSWORD')
    host, port, db_name = os.getenv('DB_HOST'), os.getenv('DB_PORT'), os.getenv('DB_NAME')
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
    engine = create_engine(connection_string)

    master_df = create_master_table_from_db(engine)

    # Load the extra geolocation dataset
    geo_df = pd.read_csv('data/raw/olist_geolocation_dataset.csv')

    # Create features for the propensity model
    modeling_df = create_propensity_features(master_df, geo_df)

    modeling_data_path = 'data/processed/propensity_dataset_advanced.csv'
    modeling_df.to_csv(modeling_data_path, index=False)
    print(f"Propensity modeling dataset with advanced features saved to {modeling_data_path}")

if __name__ == "__main__":
    main()