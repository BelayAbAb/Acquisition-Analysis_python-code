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

def basic_metrics_analysis():
    """
    Fetches data from PostgreSQL, calculates basic metrics, and saves the results as a JPEG image.
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

        # Calculate basic metrics
        metrics = {
            'Total Session Duration': {
                'mean': df['Total Session Duration'].mean(),
                'median': df['Total Session Duration'].median(),
                'std_dev': df['Total Session Duration'].std()
            },
            'Total DL (Bytes)': {
                'mean': df['total_dl_bytes'].mean(),
                'median': df['total_dl_bytes'].median(),
                'std_dev': df['total_dl_bytes'].std()
            },
            'Total UL (Bytes)': {
                'mean': df['total_ul_bytes'].mean(),
                'median': df['total_ul_bytes'].median(),
                'std_dev': df['total_ul_bytes'].std()
            }
        }

        # Create a plot for metrics summary
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, (metric_name, values) in enumerate(metrics.items()):
            ax.text(0.1, 0.9 - i * 0.2, f"{metric_name}:", fontsize=12, fontweight='bold')
            ax.text(0.1, 0.85 - i * 0.2, f"Mean: {values['mean']:.2f}", fontsize=12)
            ax.text(0.1, 0.8 - i * 0.2, f"Median: {values['median']:.2f}", fontsize=12)
            ax.text(0.1, 0.75 - i * 0.2, f"Standard Deviation: {values['std_dev']:.2f}", fontsize=12)

        # Add an explanation of the metrics
        explanation = (
            "Mean: The average value of the metric, indicating the central tendency.\n"
            "Median: The middle value, which is less affected by outliers.\n"
            "Standard Deviation: Measures the amount of variation or dispersion from the mean.\n\n"
            "These metrics help in understanding the general trends and variability in the data."
        )
        ax.text(0.1, 0.1, explanation, fontsize=10, bbox=dict(facecolor='lightgray', alpha=0.5))

        # Remove axes
        ax.axis('off')

        # Save the plot as a JPEG file
        plt.tight_layout()
        plt.savefig('basic_metrics_analysis.jpg', format='jpg')
        plt.close()

        print("Basic metrics analysis results saved to 'basic_metrics_analysis.jpg'")
    else:
        print("Failed to retrieve data.")

# Execute the function to perform basic metrics analysis
if __name__ == "__main__":
    basic_metrics_analysis()

