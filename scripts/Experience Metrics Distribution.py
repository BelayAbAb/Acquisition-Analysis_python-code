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

def analyze_metrics_by_handset_type(df):
    """
    Analyze the distribution of throughput and TCP retransmission metrics by handset type.
    """
    if df is not None:
        # Ensure necessary columns exist
        required_columns = ['Handset Type', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)', 
                            'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Missing columns for analysis: {missing_columns}")
            return
        
        # Aggregate metrics by handset type
        metrics_by_handset = df.groupby('Handset Type').agg(
            avg_throughput_dl=pd.NamedAgg(column='Avg Bearer TP DL (kbps)', aggfunc='mean'),
            avg_throughput_ul=pd.NamedAgg(column='Avg Bearer TP UL (kbps)', aggfunc='mean'),
            avg_tcp_dl_retrans=pd.NamedAgg(column='TCP DL Retrans. Vol (Bytes)', aggfunc='mean'),
            avg_tcp_ul_retrans=pd.NamedAgg(column='TCP UL Retrans. Vol (Bytes)', aggfunc='mean')
        ).reset_index()

        # Ensure the output directory exists
        output_dir = 'metrics_by_handset_type'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Plot Throughput Distribution by Handset Type
        fig, axs = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
        fig.suptitle('Throughput and TCP Retransmission by Handset Type', fontsize=16)

        # Plot Average Throughput Downlink
        axs[0, 0].bar(metrics_by_handset['Handset Type'], metrics_by_handset['avg_throughput_dl'], color='skyblue', edgecolor='black')
        axs[0, 0].set_title('Average Throughput DL (kbps)')
        axs[0, 0].set_xlabel('Handset Type')
        axs[0, 0].set_ylabel('Average Throughput DL (kbps)')
        axs[0, 0].tick_params(axis='x', rotation=90)
        axs[0, 0].grid(True)

        # Plot Average Throughput Uplink
        axs[0, 1].bar(metrics_by_handset['Handset Type'], metrics_by_handset['avg_throughput_ul'], color='salmon', edgecolor='black')
        axs[0, 1].set_title('Average Throughput UL (kbps)')
        axs[0, 1].set_xlabel('Handset Type')
        axs[0, 1].set_ylabel('Average Throughput UL (kbps)')
        axs[0, 1].tick_params(axis='x', rotation=90)
        axs[0, 1].grid(True)

        # Plot Average TCP DL Retransmission
        axs[1, 0].bar(metrics_by_handset['Handset Type'], metrics_by_handset['avg_tcp_dl_retrans'], color='lightgreen', edgecolor='black')
        axs[1, 0].set_title('Average TCP DL Retransmission (Bytes)')
        axs[1, 0].set_xlabel('Handset Type')
        axs[1, 0].set_ylabel('Average TCP DL Retransmission (Bytes)')
        axs[1, 0].tick_params(axis='x', rotation=90)
        axs[1, 0].grid(True)

        # Plot Average TCP UL Retransmission
        axs[1, 1].bar(metrics_by_handset['Handset Type'], metrics_by_handset['avg_tcp_ul_retrans'], color='orange', edgecolor='black')
        axs[1, 1].set_title('Average TCP UL Retransmission (Bytes)')
        axs[1, 1].set_xlabel('Handset Type')
        axs[1, 1].set_ylabel('Average TCP UL Retransmission (Bytes)')
        axs[1, 1].tick_params(axis='x', rotation=90)
        axs[1, 1].grid(True)

        # Save the grid plot as a JPG file
        plt.savefig(os.path.join(output_dir, 'metrics_by_handset_type.jpg'), format='jpg')
        plt.close()
        
        # Print results
        print("Metrics by Handset Type:")
        print(metrics_by_handset)
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
    
    # Perform analysis by handset type
    analyze_metrics_by_handset_type(df)
    
    # Additional analysis functions can be added here
    # For example: optimize_kmeans_clustering(df), pca_analysis(df)
