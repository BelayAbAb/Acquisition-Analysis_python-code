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

def get_top_10_handsets():
    """
    Fetches the top 10 most popular handsets from the PostgreSQL database and saves the result as a JPG image.
    """
    query = """
    SELECT
        "Handset Type",
        COUNT(*) AS usage_count
    FROM
        public.xdr_data
    GROUP BY
        "Handset Type"
    ORDER BY
        usage_count DESC
    LIMIT 10;
    """
   
    # Use the preferred function to load data
    df = load_data_using_sqlalchemy(query)
    # Or use load_data_from_postgres(query) if preferred

    if df is not None and not df.empty:
        # Clean and preprocess the data
        df["Handset Type"] = df["Handset Type"].fillna("Unknown")
        df["usage_count"] = pd.to_numeric(df["usage_count"], errors='coerce').fillna(0).astype(int)

        try:
            # Plotting
            plt.figure(figsize=(10, 6))
            plt.bar(df["Handset Type"], df["usage_count"], color='skyblue')
            plt.xlabel('Handset Type')
            plt.ylabel('Usage Count')
            plt.title('Top 10 Most Popular Handsets')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels

            # Save the plot as a JPG image
            plt.savefig('top_10_handsets.jpg', format='jpg')
            plt.close()

            print("Top 10 handsets plot saved as 'top_10_handsets.jpg'.")
       
        except Exception as e:
            print(f"An error occurred while plotting: {e}")

    else:
        print("Failed to retrieve data or the data is empty.")

# Execute the function to get top 10 handsets
if __name__ == "__main__":
    get_top_10_handsets()

