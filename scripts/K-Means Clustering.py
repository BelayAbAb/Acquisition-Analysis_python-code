import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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

def preprocess_data(df):
    """
    Preprocess the data by handling missing values and converting columns to appropriate types.
    """
    # Handle missing values
    df = df.fillna(0)
    
    # Convert categorical columns to numerical if needed
    # For simplicity, we'll use a placeholder for conversion in this example
    # Example: df['Handset Manufacturer'] = df['Handset Manufacturer'].astype('category').cat.codes
    
    return df

def cluster_and_describe(df):
    """
    Perform clustering on the data and describe each cluster based on performance metrics.
    """
    if df is not None:
        # Ensure necessary columns exist
        required_columns = ['Dur. (ms)', 'Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)', 'Total DL (Bytes)', 'Total UL (Bytes)']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Missing columns for clustering: {missing_columns}")
            return

        # Select relevant columns
        data = df[required_columns]

        # Normalize the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        # Determine the optimal number of clusters
        inertias = []
        silhouette_scores = []
        k_range = range(1, 11)  # Check for k from 1 to 10

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(data_scaled)
            inertia = kmeans.inertia_
            inertias.append(inertia)
            
            if k > 1:
                labels = kmeans.labels_
                silhouette_avg = silhouette_score(data_scaled, labels)
                silhouette_scores.append(silhouette_avg)

        # Save Elbow Method and Silhouette Scores plots
        plt.figure(figsize=(12, 6))
        plt.plot(k_range, inertias, marker='o', linestyle='--')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.grid(True)
        plt.savefig('clustering_results/elbow_method.jpg', format='jpg')
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--')
        plt.title('Silhouette Score for Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.grid(True)
        plt.savefig('clustering_results/silhouette_scores.jpg', format='jpg')
        plt.close()

        # Perform clustering with the optimal number of clusters (e.g., k=3)
        optimal_k = 3  # Replace with actual optimal k based on the plots
        kmeans = KMeans(n_clusters=optimal_k, random_state=0)
        labels = kmeans.fit_predict(data_scaled)
        df['Cluster'] = labels

        # Describe each cluster
        cluster_descriptions = df.groupby('Cluster').agg({
            'Dur. (ms)': ['mean', 'std'],
            'Avg RTT DL (ms)': ['mean', 'std'],
            'Avg RTT UL (ms)': ['mean', 'std'],
            'Avg Bearer TP DL (kbps)': ['mean', 'std'],
            'Avg Bearer TP UL (kbps)': ['mean', 'std'],
            'Total DL (Bytes)': ['mean', 'std'],
            'Total UL (Bytes)': ['mean', 'std']
        })

        # Save cluster descriptions to Excel
        with pd.ExcelWriter('clustering_results/cluster_descriptions.xlsx') as writer:
            cluster_descriptions.to_excel(writer, sheet_name='Cluster Descriptions')

        print("\nCluster Descriptions saved as 'clustering_results/cluster_descriptions.xlsx'")

    else:
        print("Failed to retrieve data.")

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
    
    # Preprocess the data
    df = preprocess_data(df)
    
    # Ensure the output directory for clustering results exists
    if not os.path.exists('clustering_results'):
        os.makedirs('clustering_results')
    
    # Perform clustering and describe each cluster
    cluster_and_describe(df)
    
    # Additional analysis functions can be added here
