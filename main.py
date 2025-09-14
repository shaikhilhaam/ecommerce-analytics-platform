# main.py
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
from src.data_processing import create_master_table_from_db

def main():
    """Main function to run the data processing pipeline from the database."""
    # Load environment variables to get DB credentials
    load_dotenv()
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    host = os.getenv('DB_HOST')
    port = os.getenv('DB_PORT')
    db_name = os.getenv('DB_NAME')
    
    # Create the database connection engine
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
    engine = create_engine(connection_string)

    # Define path for the output file
    processed_data_path = 'data/processed/master_dataset.csv'

    # Step 1: Create master table by querying the database
    master_df = create_master_table_from_db(engine)

    # Step 2: Save the processed data
    master_df.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to {processed_data_path}")

if __name__ == "__main__":
    main()