import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

def bivariate_analysis(df):
    """
    Performs bivariate analysis to explore relationships between application usage and total data (DL+UL).
    """
    if df is not None:
        # Define columns to analyze
        variables = [
            'Social Media DL (Bytes)',
            'Social Media UL (Bytes)',
            'Youtube DL (Bytes)',
            'Youtube UL (Bytes)',
            'Netflix DL (Bytes)',
            'Netflix UL (Bytes)',
            'Google DL (Bytes)',
            'Google UL (Bytes)',
            'Email DL (Bytes)',
            'Email UL (Bytes)',
            'Gaming DL (Bytes)',
            'Gaming UL (Bytes)',
            'Other DL (Bytes)',
            'Other UL (Bytes)'
        ]

        # Data for analysis
        total_data = df['Total DL (Bytes)'] + df['Total UL (Bytes)']
       
        # Create a directory for saving plots if it doesn't exist
        if not os.path.exists('bivariate_plots'):
            os.makedirs('bivariate_plots')

        # Prepare a 4x4 grid for plots
        num_vars = len(variables)
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

        for i, var in enumerate(variables):
            if var in df.columns:
                ax = axes[i]
                ax.scatter(total_data, df[var], alpha=0.5)
                ax.set_title(f'{var}')
                ax.set_xlabel('Total Data (DL+UL)')
                ax.set_ylabel(var)
                ax.grid(True)
            else:
                # Hide unused subplots
                axes[i].axis('off')

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig('bivariate_plots/bivariate_analysis_grid.jpg', format='jpg')
        plt.close()

        print("Bivariate analysis plots saved to 'bivariate_plots/bivariate_analysis_grid.jpg'.")
    else:
        print("Failed to retrieve data.")

def main():
    """
    Main function to load data, perform bivariate analysis, and save results.
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

    # Perform bivariate analysis
    bivariate_analysis(df)

# Execute the main function
if __name__ == "__main__":
    main()

