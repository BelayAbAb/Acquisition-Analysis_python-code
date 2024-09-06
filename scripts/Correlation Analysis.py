import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns

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

def correlation_analysis(df):
    """
    Performs correlation analysis between specific data columns and saves the correlation matrix as an image.
    """
    if df is not None:
        # Define the columns to analyze
        columns = [
            'Social Media DL (Bytes)',
            'Social Media UL (Bytes)',
            'Google DL (Bytes)',
            'Google UL (Bytes)',
            'Email DL (Bytes)',
            'Email UL (Bytes)',
            'Youtube DL (Bytes)',
            'Youtube UL (Bytes)',
            'Netflix DL (Bytes)',
            'Netflix UL (Bytes)',
            'Gaming DL (Bytes)',
            'Gaming UL (Bytes)',
            'Other DL (Bytes)',
            'Other UL (Bytes)'
        ]

        # Ensure columns exist in the DataFrame
        available_columns = [col for col in columns if col in df.columns]
       
        if len(available_columns) < 2:
            print("Not enough columns available for correlation analysis.")
            return

        # Calculate the correlation matrix
        corr_matrix = df[available_columns].corr()

        # Create a directory for saving the correlation matrix image if it doesn't exist
        if not os.path.exists('correlation_matrix'):
            os.makedirs('correlation_matrix')

        # Plot the correlation matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.savefig('correlation_matrix/correlation_matrix.jpg', format='jpg')
        plt.close()

        # Print correlation matrix
        print("Correlation Matrix:")
        print(corr_matrix)
        print("Correlation matrix saved to 'correlation_matrix/correlation_matrix.jpg'.")

        # Interpretation
        interpret_correlation(corr_matrix)
    else:
        print("Failed to retrieve data.")

def interpret_correlation(corr_matrix):
    """
    Provides interpretation of the correlation matrix.
    """
    print("\nInterpretation of Correlation Matrix:")
    for col in corr_matrix.columns:
        for row in corr_matrix.index:
            if row != col:
                correlation = corr_matrix.at[row, col]
                if abs(correlation) > 0.7:
                    strength = "strong"
                elif abs(correlation) > 0.5:
                    strength = "moderate"
                elif abs(correlation) > 0.3:
                    strength = "weak"
                else:
                    strength = "very weak"
               
                direction = "positive" if correlation > 0 else "negative"
                print(f"Correlation between {row} and {col}: {direction} ({strength}) with a value of {correlation:.2f}")

def main():
    """
    Main function to load data, perform correlation analysis, and save results.
    """
    # SQL query to fetch relevant data
    query = """
    SELECT
        "Social Media DL (Bytes)",
        "Social Media UL (Bytes)",
        "Youtube DL (Bytes)",
        "Youtube UL (Bytes)",
        "Netflix DL (Bytes)",
        "Netflix UL (Bytes)",
        "Google DL (Bytes)",
        "Google UL (Bytes)",
        "Email DL (Bytes)",
        "Email UL (Bytes)",
        "Gaming DL (Bytes)",
        "Gaming UL (Bytes)",
        "Other DL (Bytes)",
        "Other UL (Bytes)",
        "Total DL (Bytes)",
        "Total UL (Bytes)"
    FROM
        public.xdr_data
    """
   
    # Load data
    df = load_data_using_sqlalchemy(query)
    # Alternatively: df = load_data_from_postgres(query)

    # Perform correlation analysis
    correlation_analysis(df)

# Execute the main function
if __name__ == "__main__":
    main()