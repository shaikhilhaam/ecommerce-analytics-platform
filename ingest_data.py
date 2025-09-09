import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Get database credentials from environment variables
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    host = os.getenv('DB_HOST')
    port = os.getenv('DB_PORT')
    db_name = os.getenv('DB_NAME')

    # Create the database connection string
    # This format is specific to PostgreSQL
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"

    # Create the SQLAlchemy engine
    engine = create_engine(connection_string)
    print("Successfully connected to the database.")

    # Path to the directory containing the CSV files
    data_dir = os.path.join('data', 'raw')

    # Get a list of all CSV files in the directory
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    # Loop through each CSV file and load it into the database
    for file_name in csv_files:
        # Construct the full file path
        file_path = os.path.join(data_dir, file_name)

        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)
        
        # Create a clean table name from the file name
        # Example: 'olist_customers_dataset.csv' becomes 'olist_customers'
        table_name = file_name.replace('_dataset.csv', '').replace('.csv', '')
        
        print(f"Loading data from {file_name} into table {table_name}...")

        # Use the to_sql method to write the DataFrame to a SQL table
        # - 'if_exists='replace'' will drop the table if it already exists and create a new one.
        # - 'index=False' prevents pandas from writing the DataFrame index as a column.
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        
        print(f"Successfully loaded {table_name}.")

    print("All data has been successfully ingested into the database.")

if __name__ == '__main__':
    main()