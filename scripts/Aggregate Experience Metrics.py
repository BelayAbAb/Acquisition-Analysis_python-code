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
    return df

def aggregate_experience_metrics(df):
    """
    Calculate average TCP retransmission, RTT, throughput, and handset types per user.
    Compute top, bottom, and frequent values, and save results in JPG format.
    """
    if df is not None:
        # Ensure necessary columns exist
        required_columns = ['IMSI', 'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)', 
                            'Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 
                            'Avg Bearer TP UL (kbps)', 'Handset Type']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Missing columns for aggregation: {missing_columns}")
            return
        
        # Aggregate metrics per user
        aggregated_df = df.groupby('IMSI').agg(
            avg_tcp_dl_retrans=pd.NamedAgg(column='TCP DL Retrans. Vol (Bytes)', aggfunc='mean'),
            avg_tcp_ul_retrans=pd.NamedAgg(column='TCP UL Retrans. Vol (Bytes)', aggfunc='mean'),
            avg_rtt_dl=pd.NamedAgg(column='Avg RTT DL (ms)', aggfunc='mean'),
            avg_rtt_ul=pd.NamedAgg(column='Avg RTT UL (ms)', aggfunc='mean'),
            avg_throughput_dl=pd.NamedAgg(column='Avg Bearer TP DL (kbps)', aggfunc='mean'),
            avg_throughput_ul=pd.NamedAgg(column='Avg Bearer TP UL (kbps)', aggfunc='mean'),
            handset_type=pd.NamedAgg(column='Handset Type', aggfunc=lambda x: x.mode()[0] if not x.mode().empty else 'Unknown')
        ).reset_index()

        # Calculate top, bottom, and frequent values for each metric
        metrics_summary = {}
        
        # For each metric
        for column in ['avg_tcp_dl_retrans', 'avg_tcp_ul_retrans', 'avg_rtt_dl', 'avg_rtt_ul', 
                       'avg_throughput_dl', 'avg_throughput_ul']:
            metrics_summary[column] = {
                'Top Value': aggregated_df[column].max(),
                'Bottom Value': aggregated_df[column].min(),
                'Frequent Value': aggregated_df[column].mode()[0] if not aggregated_df[column].mode().empty else 'Unknown'
            }
        
        # Handset Type Summary
        handset_summary = aggregated_df['handset_type'].value_counts()

        # Ensure the output directory exists
        output_dir = 'metrics_results'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Plot metrics in a 2x3 grid
        fig, axs = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
        fig.suptitle('Metrics Distributions', fontsize=16)

        metrics = ['avg_tcp_dl_retrans', 'avg_tcp_ul_retrans', 'avg_rtt_dl', 'avg_rtt_ul', 
                   'avg_throughput_dl', 'avg_throughput_ul']
        
        for ax, column in zip(axs.flatten(), metrics):
            ax.hist(aggregated_df[column], bins=30, color='skyblue', edgecolor='black')
            ax.set_title(f'{column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')
            ax.grid(True)

        # Save the grid plot as a JPG file
        plt.savefig(os.path.join(output_dir, 'metrics_distribution_grid.jpg'), format='jpg')
        plt.close()
        
        # Plot handset type distribution
        plt.figure(figsize=(12, 8))
        handset_summary.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('Handset Type Distribution')
        plt.xlabel('Handset Type')
        plt.ylabel('Frequency')
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'handset_type_distribution.jpg'), format='jpg')
        plt.close()
        
        # Print results
        print("Aggregated Experience Metrics:")
        print(aggregated_df.head())
        
        print("\nMetric Summary:")
        for metric, summary in metrics_summary.items():
            print(f"{metric}:")
            print(f"  Top Value: {summary['Top Value']}")
            print(f"  Bottom Value: {summary['Bottom Value']}")
            print(f"  Frequent Value: {summary['Frequent Value']}")
            print()
        
        print("\nHandset Type Summary:")
        print(handset_summary)
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
    
    # Perform aggregation and analysis
    aggregate_experience_metrics(df)
    
    # Additional analysis functions can be added here
    # For example: optimize_kmeans_clustering(df), pca_analysis(df)
