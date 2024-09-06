import os
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

def calculate_metrics_per_user():
    """
    Calculates metrics per user from the PostgreSQL database.
    """
    query = """
    SELECT
        "IMSI",
        COUNT(*) AS number_of_sessions,
        SUM("Dur. (ms)") AS total_session_duration,
        SUM("Total DL (Bytes)") AS total_download_data,
        SUM("Total UL (Bytes)") AS total_upload_data,
        SUM("Social Media DL (Bytes)") AS social_media_dl,
        SUM("Google DL (Bytes)") AS google_dl,
        SUM("Email DL (Bytes)") AS email_dl,
        SUM("Youtube DL (Bytes)") AS youtube_dl,
        SUM("Netflix DL (Bytes)") AS netflix_dl,
        SUM("Gaming DL (Bytes)") AS gaming_dl,
        SUM("Other DL (Bytes)") AS other_dl
    FROM
        public.xdr_data
    GROUP BY
        "IMSI";
    """
   
    df = load_data_using_sqlalchemy(query)

    if df is not None and not df.empty:
        # Clean and preprocess the data
        df["number_of_sessions"] = df["number_of_sessions"].astype(int)
        df["total_session_duration"] = pd.to_numeric(df["total_session_duration"], errors='coerce').fillna(0).astype(float)
        df["total_download_data"] = pd.to_numeric(df["total_download_data"], errors='coerce').fillna(0).astype(float)
        df["total_upload_data"] = pd.to_numeric(df["total_upload_data"], errors='coerce').fillna(0).astype(float)
        df["social_media_dl"] = pd.to_numeric(df["social_media_dl"], errors='coerce').fillna(0).astype(float)
        df["google_dl"] = pd.to_numeric(df["google_dl"], errors='coerce').fillna(0).astype(float)
        df["email_dl"] = pd.to_numeric(df["email_dl"], errors='coerce').fillna(0).astype(float)
        df["youtube_dl"] = pd.to_numeric(df["youtube_dl"], errors='coerce').fillna(0).astype(float)
        df["netflix_dl"] = pd.to_numeric(df["netflix_dl"], errors='coerce').fillna(0).astype(float)
        df["gaming_dl"] = pd.to_numeric(df["gaming_dl"], errors='coerce').fillna(0).astype(float)
        df["other_dl"] = pd.to_numeric(df["other_dl"], errors='coerce').fillna(0).astype(float)

        # Save the results to a CSV file
        df.to_csv('user_metrics.csv', index=False)

        print("Data aggregation completed and saved to 'user_metrics.csv'.")
       
        return df
    else:
        print("Failed to retrieve data or the data is empty.")
        return pd.DataFrame()

def visualize_metrics(df):
    """
    Visualizes the metrics data from the DataFrame.
    """
    if not df.empty:
        plt.figure(figsize=(15, 10))
       
        # Plot total download data per user
        plt.subplot(2, 2, 1)
        plt.hist(df["total_download_data"], bins=30, color='skyblue', edgecolor='black')
        plt.title('Total Download Data per User')
        plt.xlabel('Bytes')
        plt.ylabel('Frequency')

        # Plot total upload data per user
        plt.subplot(2, 2, 2)
        plt.hist(df["total_upload_data"], bins=30, color='lightgreen', edgecolor='black')
        plt.title('Total Upload Data per User')
        plt.xlabel('Bytes')
        plt.ylabel('Frequency')

        # Plot total session duration per user
        plt.subplot(2, 2, 3)
        plt.hist(df["total_session_duration"], bins=30, color='salmon', edgecolor='black')
        plt.title('Total Session Duration per User')
        plt.xlabel('Milliseconds')
        plt.ylabel('Frequency')

        # Plot download data by application
        applications = ['social_media_dl', 'google_dl', 'email_dl', 'youtube_dl', 'netflix_dl', 'gaming_dl', 'other_dl']
        plt.subplot(2, 2, 4)
        df[applications].plot(kind='box', vert=False, patch_artist=True, notch=True, ax=plt.gca())
        plt.title('Download Data by Application')
        plt.xlabel('Bytes')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
       
        # Save the plots as a JPG file
        plt.savefig('user_metrics_visualization.jpg', format='jpg')
        plt.close()

        print("Visualizations saved to 'user_metrics_visualization.jpg'.")
    else:
        print("No data available for visualization.")

# Execute the functions to calculate and visualize metrics
if __name__ == "__main__":
    df = calculate_metrics_per_user()
    visualize_metrics(df)
