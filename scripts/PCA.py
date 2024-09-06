import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import psycopg2

# Load environment variables from .env file
load_dotenv()

def get_database_connection():
    """
    Create and return a connection to the PostgreSQL database using environment variables.
    """
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
    return conn

def load_data(query):
    """
    Load data from the PostgreSQL database into a DataFrame.
    """
    conn = get_database_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def pca_analysis(df):
    """
    Performs PCA on the DataFrame and provides interpretation of the results.
    """
    if df is not None:
        # Define the columns for PCA
        columns = [
            'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
            'Google DL (Bytes)', 'Google UL (Bytes)',
            'Email DL (Bytes)', 'Email UL (Bytes)',
            'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
            'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
            'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
            'Other DL (Bytes)', 'Other UL (Bytes)'
        ]
        
        # Ensure columns exist in the DataFrame
        available_columns = [col for col in columns if col in df.columns]
        
        if len(available_columns) < 2:
            print("Not enough columns available for PCA analysis.")
            print("Available columns for PCA:", available_columns)
            return
        
        # Drop rows with NaN values in the selected columns
        X = df[available_columns].dropna()
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # PCA results
        explained_variance = pca.explained_variance_ratio_
        cumulative_explained_variance = explained_variance.cumsum()
        
        # Ensure the pca_results directory exists
        output_dir = 'pca_results'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Plot explained variance
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
        plt.title('Cumulative Explained Variance by Principal Components')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'cumulative_explained_variance.jpg'), format='jpg')
        plt.close()
        
        # Print PCA results
        print("\nPCA Results:")
        print(f"Explained variance ratio for each component: {explained_variance}")
        print(f"Cumulative explained variance: {cumulative_explained_variance}")
        
        # Interpretation
        interpret_pca(explained_variance, cumulative_explained_variance)
    else:
        print("Failed to retrieve data.")

def interpret_pca(explained_variance, cumulative_explained_variance):
    """
    Provides interpretation of PCA results.
    """
    print("\nInterpretation of PCA Results:")
    num_components = len(explained_variance)
    significant_components = [i + 1 for i in range(num_components) if cumulative_explained_variance[i] > 0.9]
    
    print(f"1. The first few principal components capture a significant portion of the variance.")
    print(f"2. The cumulative explained variance by the first {len(significant_components)} components is above 90%.")
    print(f"3. The most important principal components can be used to reduce the dimensionality of the data while retaining most of the information.")
    print(f"4. By focusing on these principal components, you can simplify the analysis and visualization without losing much information.")

    if len(significant_components) == 0:
        print("5. No principal components capture more than 90% of the variance. Consider evaluating more components or revisiting the data preprocessing.")

if __name__ == "__main__":
    # Define the SQL query to fetch data
    query = """
    SELECT "Bearer Id", "Start", "Start ms", "End", "End ms", "Dur. (ms)", "IMSI", "MSISDN/Number", "IMEI", "Last Location Name",
           "Avg RTT DL (ms)", "Avg RTT UL (ms)", "Avg Bearer TP DL (kbps)", "Avg Bearer TP UL (kbps)",
           "TCP DL Retrans. Vol (Bytes)", "TCP UL Retrans. Vol (Bytes)", "DL TP < 50 Kbps (%)", "50 Kbps < DL TP < 250 Kbps (%)",
           "250 Kbps < DL TP < 1 Mbps (%)", "DL TP > 1 Mbps (%)", "UL TP < 10 Kbps (%)", "10 Kbps < UL TP < 50 Kbps (%)",
           "50 Kbps < UL TP < 300 Kbps (%)", "UL TP > 300 Kbps (%)", "HTTP DL (Bytes)", "HTTP UL (Bytes)",
           "Activity Duration DL (ms)", "Activity Duration UL (ms)", "Dur. (ms).1", "Handset Manufacturer", "Handset Type",
           "Nb of sec with 125000B < Vol DL", "Nb of sec with 1250B < Vol UL < 6250B", "Nb of sec with 31250B < Vol DL < 125000B",
           "Nb of sec with 37500B < Vol UL", "Nb of sec with 6250B < Vol DL < 31250B", "Nb of sec with 6250B < Vol UL < 37500B",
           "Nb of sec with Vol DL < 6250B", "Nb of sec with Vol UL < 1250B", "Social Media DL (Bytes)", "Social Media UL (Bytes)",
           "Google DL (Bytes)", "Google UL (Bytes)", "Email DL (Bytes)", "Email UL (Bytes)", "Youtube DL (Bytes)",
           "Youtube UL (Bytes)", "Netflix DL (Bytes)", "Netflix UL (Bytes)", "Gaming DL (Bytes)", "Gaming UL (Bytes)",
           "Other DL (Bytes)", "Other UL (Bytes)", "Total UL (Bytes)", "Total DL (Bytes)"
    FROM public.xdr_data
    """
    
    # Load data from the database
    df = load_data(query)
    
    # Perform PCA analysis
    pca_analysis(df)
