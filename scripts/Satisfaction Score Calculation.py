import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
    print(f"Data loaded with {len(df)} rows and columns: {list(df.columns)}")
    return df

def calculate_satisfaction_scores(df):
    """
    Calculate engagement and experience scores and determine overall satisfaction scores.
    Identify the top satisfied customers based on their overall satisfaction scores.
    """
    if df is not None:
        # Ensure necessary columns exist
        required_columns = ['IMSI', 'Dur. (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)', 'Avg RTT DL (ms)', 'Avg RTT UL (ms)']
        missing_columns = [col for col in required_columns if col not in df.columns]
       
        if missing_columns:
            print(f"Missing columns for satisfaction calculation: {missing_columns}")
            return
       
        # Aggregate metrics for engagement and experience
        metrics_df = df.groupby('IMSI').agg(
            total_duration=pd.NamedAgg(column='Dur. (ms)', aggfunc='sum'),
            total_dl_traffic=pd.NamedAgg(column='Total DL (Bytes)', aggfunc='sum'),
            total_ul_traffic=pd.NamedAgg(column='Total UL (Bytes)', aggfunc='sum'),
            avg_rtt_dl=pd.NamedAgg(column='Avg RTT DL (ms)', aggfunc='mean'),
            avg_rtt_ul=pd.NamedAgg(column='Avg RTT UL (ms)', aggfunc='mean')
        ).reset_index()
       
        # Normalize the metrics
        scaler = StandardScaler()
        metrics_scaled = scaler.fit_transform(metrics_df[['total_duration', 'total_dl_traffic', 'total_ul_traffic', 'avg_rtt_dl', 'avg_rtt_ul']])
       
        # Calculate engagement and experience scores
        # Define engagement score as a combination of total data traffic and duration
        metrics_df['engagement_score'] = metrics_df[['total_dl_traffic', 'total_ul_traffic', 'total_duration']].mean(axis=1)
       
        # Define experience score as the inverse of average RTT (lower is better)
        metrics_df['experience_score'] = 1 / (metrics_df[['avg_rtt_dl', 'avg_rtt_ul']].mean(axis=1) + 1)  # Added 1 to avoid division by zero
       
        # Calculate overall satisfaction score
        metrics_df['satisfaction_score'] = (metrics_df['engagement_score'] + metrics_df['experience_score']) / 2
       
        # Identify top satisfied customers
        top_customers = metrics_df.sort_values(by='satisfaction_score', ascending=False).head(10)
       
        # Print top satisfied customers
        print("\nTop 10 Satisfied Customers:")
        print(top_customers[['IMSI', 'engagement_score', 'experience_score', 'satisfaction_score']])
       
        # Optionally, save the results to a CSV file
        output_dir = 'satisfaction_results'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Directory {output_dir} created.")
       
        satisfaction_path = os.path.join(output_dir, 'satisfaction_scores.csv')
        metrics_df.to_csv(satisfaction_path, index=False)
        print(f"Satisfaction scores saved to {satisfaction_path}.")
   
    else:
        print("Failed to retrieve data.")

def optimize_kmeans_clustering(df):
    """
    Determine the optimal number of clusters for user engagement using the Elbow Method and Silhouette Score.
    """
    if df is not None:
        # Ensure necessary columns exist
        required_columns = ['IMSI', 'Dur. (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)']
        missing_columns = [col for col in required_columns if col not in df.columns]
       
        if missing_columns:
            print(f"Missing columns for clustering: {missing_columns}")
            return
       
        # Aggregate metrics
        metrics_df = df.groupby('IMSI').agg(
            total_duration=pd.NamedAgg(column='Dur. (ms)', aggfunc='sum'),
            total_dl_traffic=pd.NamedAgg(column='Total DL (Bytes)', aggfunc='sum'),
            total_ul_traffic=pd.NamedAgg(column='Total UL (Bytes)', aggfunc='sum')
        ).reset_index()
       
        # Normalize the metrics
        scaler = StandardScaler()
        metrics_scaled = scaler.fit_transform(metrics_df[['total_duration', 'total_dl_traffic', 'total_ul_traffic']])
       
        # Elbow Method
        inertias = []
        silhouette_scores = []
        k_range = range(1, 11)  # Check for k from 1 to 10
       
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(metrics_scaled)
            inertia = kmeans.inertia_
            inertias.append(inertia)
           
            if k > 1:
                labels = kmeans.labels_
                silhouette_avg = silhouette_score(metrics_scaled, labels)
                silhouette_scores.append(silhouette_avg)
       
        # Ensure the output directory exists
        output_dir = 'clustering_results'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Directory {output_dir} created.")
        else:
            print(f"Directory {output_dir} already exists.")
       
        # Plot Elbow Method
        plt.figure(figsize=(12, 6))
        plt.plot(k_range, inertias, marker='o', linestyle='--')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.grid(True)
        elbow_path = os.path.join(output_dir, 'elbow_method.jpg')
        plt.savefig(elbow_path, format='jpg')
        print(f"Elbow method plot saved to {elbow_path}.")
        plt.close()
       
        # Plot Silhouette Scores
        plt.figure(figsize=(12, 6))
        plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--')
        plt.title('Silhouette Score for Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.grid(True)
        silhouette_path = os.path.join(output_dir, 'silhouette_scores.jpg')
        plt.savefig(silhouette_path, format='jpg')
        print(f"Silhouette scores plot saved to {silhouette_path}.")
        plt.close()
       
        # Print results
        print("\nElbow Method Results:")
        print(f"Inertias for k=1 to 10: {inertias}")
       
        print("\nSilhouette Scores Results:")
        print(f"Silhouette Scores for k=2 to 10: {silhouette_scores}")
       
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
   
    # Ensure the output directory for clustering results exists
    if not os.path.exists('clustering_results'):
        os.makedirs('clustering_results')
   
    # Perform K-Means clustering optimization
    optimize_kmeans_clustering(df)
   
    # Perform satisfaction score calculation
    calculate_satisfaction_scores(df)
   
    # Additional analysis functions can be added here
    # For example: pca_analysis(df)
