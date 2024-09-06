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

def print_dataframe_columns(df):
    """
    Prints the columns of the DataFrame to help diagnose issues.
    """
    print("DataFrame columns:")
    print(df.columns)

def non_graphical_univariate_analysis(df):
    """
    Computes and interprets dispersion parameters (variance and standard deviation) for the data.
    """
    if df is not None:
        # Print DataFrame columns for debugging
        print_dataframe_columns(df)

        # Compute dispersion parameters
        metrics = {
            'Total Session Duration': {
                'variance': df['duration_ms'].var(),
                'std_dev': df['duration_ms'].std()
            },
            'Total DL (Bytes)': {
                'variance': df['total_dl_bytes'].var(),
                'std_dev': df['total_dl_bytes'].std()
            },
            'Total UL (Bytes)': {
                'variance': df['total_ul_bytes'].var(),
                'std_dev': df['total_ul_bytes'].std()
            }
        }

        # Print the metrics
        print("Non-Graphical Univariate Analysis Results:")
        for variable, values in metrics.items():
            print(f"\n{variable}:")
            print(f"  Variance: {values['variance']:.2f}")
            print(f"  Standard Deviation: {values['std_dev']:.2f}")

        # Interpretations
        interpretations = (
            "\nInterpretation of Results:\n"
            "Variance measures the dispersion of data points from the mean. Higher variance indicates more spread.\n"
            "Standard Deviation is the square root of variance and provides a measure of dispersion in the same units as the data.\n"
            "A higher standard deviation suggests more variability in the data."
        )
        print(interpretations)
    else:
        print("Failed to retrieve data.")

def graphical_univariate_analysis(df):
    """
    Creates histograms and box plots for visualizing the distribution of each variable.
    """
    if df is not None:
        # Print DataFrame columns for debugging
        print_dataframe_columns(df)

        # Define the variables to analyze
        variables = ['duration_ms', 'total_dl_bytes', 'total_ul_bytes']

        # Create a directory for saving plots if it doesn't exist
        if not os.path.exists('plots'):
            os.makedirs('plots')

        for variable in variables:
            # Histogram
            plt.figure(figsize=(12, 6))
            plt.hist(df[variable].dropna(), bins=30, edgecolor='black')
            plt.title(f'Histogram of {variable}')
            plt.xlabel(variable)
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.savefig(f'plots/histogram_{variable}.jpg')
            plt.close()

            # Box Plot
            plt.figure(figsize=(12, 6))
            plt.boxplot(df[variable].dropna(), vert=False)
            plt.title(f'Box Plot of {variable}')
            plt.xlabel(variable)
            plt.grid(True)
            plt.savefig(f'plots/boxplot_{variable}.jpg')
            plt.close()

        print("Graphical Univariate Analysis plots saved to 'plots' directory.")
    else:
        print("Failed to retrieve data.")

def main():
    """
    Main function to load data, perform analyses, and save results.
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

    # Perform non-graphical and graphical univariate analysis
    non_graphical_univariate_analysis(df)
    graphical_univariate_analysis(df)

# Execute the main function
if __name__ == "__main__":
    main()