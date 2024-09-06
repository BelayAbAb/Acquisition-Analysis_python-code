import os
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Load environment variables from .env file
load_dotenv()

# Fetch database connection parameters from environment variables
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

def load_data_from_postgres(query):
    """
    Connects to the PostgreSQL database and loads data based on the provided SQL query.

    :param query: SQL query to execute.
    :return: DataFrame containing the results of the query.
    """
    try:
        # Establish a connection to the database
        connection = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )

        # Load data using pandas
        df = pd.read_sql_query(query, connection)

        # Close the database connection
        connection.close()

        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def load_data_using_sqlalchemy(query):
    """
    Connects to the PostgreSQL database and loads data based on the provided SQL query using SQLAlchemy.

    :param query: SQL query to execute.
    :return: DataFrame containing the results of the query.
    """
    try:
        # Create a connection string
        connection_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

        # Create an SQLAlchemy engine
        engine = create_engine(connection_string)

        # Load data into a pandas DataFrame
        df = pd.read_sql_query(query, engine)

        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def get_top_3_handset_manufacturers():
    """
    Fetches the top 3 handset manufacturers from the PostgreSQL database.
    """
    query = """
    SELECT
        "Handset Manufacturer",
        COUNT(*) AS usage_count
    FROM
        public.xdr_data
    GROUP BY
        "Handset Manufacturer"
    ORDER BY
        usage_count DESC
    LIMIT 3;
    """
   
    df = load_data_using_sqlalchemy(query)

    if df is not None and not df.empty:
        # Clean and preprocess the data
        df["Handset Manufacturer"] = df["Handset Manufacturer"].fillna("Unknown")
        df["usage_count"] = pd.to_numeric(df["usage_count"], errors='coerce').fillna(0).astype(int)
       
        return df["Handset Manufacturer"].tolist()
    else:
        print("Failed to retrieve data or the data is empty.")
        return []

def get_top_5_handsets_per_manufacturer(manufacturer):
    """
    Fetches the top 5 handsets for a specific manufacturer from the PostgreSQL database.
    """
    query = f"""
    SELECT
        "Handset Type",
        COUNT(*) AS usage_count
    FROM
        public.xdr_data
    WHERE
        "Handset Manufacturer" = '{manufacturer}'
    GROUP BY
        "Handset Type"
    ORDER BY
        usage_count DESC
    LIMIT 5;
    """
   
    df = load_data_using_sqlalchemy(query)

    if df is not None and not df.empty:
        # Clean and preprocess the data
        df["Handset Type"] = df["Handset Type"].fillna("Unknown")
        df["usage_count"] = pd.to_numeric(df["usage_count"], errors='coerce').fillna(0).astype(int)
       
        return df
    else:
        print(f"Failed to retrieve data for manufacturer: {manufacturer} or the data is empty.")
        return pd.DataFrame()

def visualize_top_5_handsets():
    """
    Retrieves and visualizes the top 5 handsets for each of the top 3 manufacturers.
    """
    top_manufacturers = get_top_3_handset_manufacturers()

    if top_manufacturers:
        for manufacturer in top_manufacturers:
            df = get_top_5_handsets_per_manufacturer(manufacturer)

            if not df.empty:
                plt.figure(figsize=(10, 6))
                plt.bar(df["Handset Type"], df["usage_count"], color='skyblue')
                plt.xlabel('Handset Type')
                plt.ylabel('Usage Count')
                plt.title(f'Top 5 Handsets for {manufacturer}')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels

                # Save the plot as a JPG image
                file_name = f'top_5_handsets_{manufacturer.replace(" ", "_")}.jpg'
                plt.savefig(file_name, format='jpg')
                plt.close()

                print(f"Top 5 handsets plot for {manufacturer} saved as '{file_name}'.")
            else:
                print(f"No data available for manufacturer: {manufacturer}")
    else:
        print("No top manufacturers found.")

# Execute the function to visualize top 5 handsets per top manufacturer
if __name__ == "__main__":
    visualize_top_5_handsets()