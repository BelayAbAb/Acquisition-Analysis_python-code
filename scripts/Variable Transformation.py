import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

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

def load_and_transform_data():
    """
    Fetches data from PostgreSQL, segments users into deciles, computes total data per decile,
    and saves the results as a JPEG image.
    """
    # SQL query to fetch relevant data
    query = """
    SELECT
        "Dur. (ms)" AS duration_ms,
        "Total DL (Bytes)" AS total_dl_bytes,
        "Total UL (Bytes)" AS total_ul_bytes
    FROM
        public.xdr_data
    """
   
    # Load data
    df = load_data_using_sqlalchemy(query)
    # Alternatively: df = load_data_from_postgres(query)

    if df is not None:
        # Compute Total Session Duration in seconds
        df['Total Session Duration'] = df['duration_ms'] / 1000  # Convert milliseconds to seconds

        # Ensure we have enough unique values to create quantiles
        if df['Total Session Duration'].nunique() > 1:
            # Segment users into deciles based on Total Session Duration
            df['Decile Class'] = pd.qcut(df['Total Session Duration'], 10, labels=False, duplicates='drop') + 1
        else:
            # Handle case where there are not enough unique values
            df['Decile Class'] = 1  # Assign all rows to the first decile if not enough unique values

        # Calculate Total Data (DL + UL)
        df['Total DL + UL (Bytes)'] = df['total_dl_bytes'] + df['total_ul_bytes']

        # Aggregate data per Decile Class
        decile_summary = df.groupby('Decile Class').agg(
            Total_DL_UL=('Total DL + UL (Bytes)', 'sum'),
            Count=('Decile Class', 'count')
        ).reset_index()

        # Create a plot
        plt.figure(figsize=(10, 6))
        plt.bar(decile_summary['Decile Class'].astype(str), decile_summary['Total_DL_UL'])
        plt.xlabel('Decile Class')
        plt.ylabel('Total Data (Bytes)')
        plt.title('Total Data by Decile Class')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot as a JPEG file
        plt.savefig('decile_summary.jpg', format='jpg')
        plt.close()

        print("Data transformation results saved to 'decile_summary.jpg'")
    else:
        print("Failed to retrieve data.")

# Execute the function to load and transform data
if __name__ == "__main__":
    load_and_transform_data()

