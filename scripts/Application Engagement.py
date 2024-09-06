import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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

def identify_engaged_users_and_apps(df):
    """
    Identify the top engaged users and most used applications, and save the results as JPG files.
    """
    if df is not None:
        # Ensure necessary columns exist
        engagement_columns = [
            'IMSI', 'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
            'Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)', 'Email UL (Bytes)',
            'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
            'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 'Other DL (Bytes)', 'Other UL (Bytes)'
        ]
        
        missing_columns = [col for col in engagement_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Missing columns for engagement analysis: {missing_columns}")
            return
        
        # Aggregate metrics for users
        user_metrics_df = df.groupby('IMSI').agg(
            total_social_media_dl=pd.NamedAgg(column='Social Media DL (Bytes)', aggfunc='sum'),
            total_social_media_ul=pd.NamedAgg(column='Social Media UL (Bytes)', aggfunc='sum'),
            total_google_dl=pd.NamedAgg(column='Google DL (Bytes)', aggfunc='sum'),
            total_google_ul=pd.NamedAgg(column='Google UL (Bytes)', aggfunc='sum'),
            total_email_dl=pd.NamedAgg(column='Email DL (Bytes)', aggfunc='sum'),
            total_email_ul=pd.NamedAgg(column='Email UL (Bytes)', aggfunc='sum'),
            total_youtube_dl=pd.NamedAgg(column='Youtube DL (Bytes)', aggfunc='sum'),
            total_youtube_ul=pd.NamedAgg(column='Youtube UL (Bytes)', aggfunc='sum'),
            total_netflix_dl=pd.NamedAgg(column='Netflix DL (Bytes)', aggfunc='sum'),
            total_netflix_ul=pd.NamedAgg(column='Netflix UL (Bytes)', aggfunc='sum'),
            total_gaming_dl=pd.NamedAgg(column='Gaming DL (Bytes)', aggfunc='sum'),
            total_gaming_ul=pd.NamedAgg(column='Gaming UL (Bytes)', aggfunc='sum'),
            total_other_dl=pd.NamedAgg(column='Other DL (Bytes)', aggfunc='sum'),
            total_other_ul=pd.NamedAgg(column='Other UL (Bytes)', aggfunc='sum')
        ).reset_index()
        
        # Calculate engagement score as sum of all download and upload metrics
        user_metrics_df['engagement_score'] = user_metrics_df.filter(like='total_').sum(axis=1)
        
        # Identify top 10 engaged users
        top_engaged_users = user_metrics_df.nlargest(10, 'engagement_score')
        
        # Aggregate application usage
        app_usage_df = df[['Social Media DL (Bytes)', 'Social Media UL (Bytes)',
                           'Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)', 'Email UL (Bytes)',
                           'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
                           'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 'Other DL (Bytes)', 'Other UL (Bytes)']].sum().reset_index()
        app_usage_df.columns = ['Application', 'Total Traffic (Bytes)']
        
        # Ensure the output directory exists
        output_dir = 'engagement_and_apps_results'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save top engaged users plot
        plt.figure(figsize=(12, 8))
        plt.bar(top_engaged_users['IMSI'].astype(str), top_engaged_users['engagement_score'], color='skyblue')
        plt.title('Top 10 Engaged Users')
        plt.xlabel('User IMSI')
        plt.ylabel('Engagement Score (Bytes)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'top_engaged_users.jpg'), format='jpg')
        plt.close()
        
        # Save most used applications plot
        plt.figure(figsize=(12, 8))
        plt.bar(app_usage_df['Application'], app_usage_df['Total Traffic (Bytes)'], color='lightcoral')
        plt.title('Most Used Applications')
        plt.xlabel('Application')
        plt.ylabel('Total Traffic (Bytes)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'most_used_applications.jpg'), format='jpg')
        plt.close()
        
        # Print results
        print("\nTop Engaged Users:")
        print(top_engaged_users[['IMSI', 'engagement_score']])
        
        print("\nMost Used Applications:")
        print(app_usage_df)
        
    else:
        print("Failed to retrieve data.")

def normalize_and_cluster(df):
    """
    Normalize metrics and perform K-Means clustering to classify users into engagement clusters.
    """
    if df is not None:
        # Ensure the necessary columns exist
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
        
        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=3, random_state=0)  # Adjust the number of clusters as needed
        metrics_df['cluster'] = kmeans.fit_predict(metrics_scaled)
        
        # Ensure the output directory exists
        output_dir = 'clustering_results'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save clustering results
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(metrics_df['total_dl_traffic'], metrics_df['total_ul_traffic'], c=metrics_df['cluster'], cmap='viridis')
        plt.colorbar(scatter, label='Cluster')
        plt.title('User Engagement Clusters')
        plt.xlabel('Total DL Traffic (Bytes)')
        plt.ylabel('Total UL Traffic (Bytes)')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'user_engagement_clusters.jpg'), format='jpg')
        plt.close()
        
        # Print cluster centers and distribution
        print("\nCluster Centers:")
        print(kmeans.cluster_centers_)
        print("\nCluster Distribution:")
        print(metrics_df['cluster'].value_counts())
        
    else:
        print("Failed to retrieve data.")

def save_top_users_grid(top_users_dict, output_dir):
    """
    Save a 2x2 grid of bar plots of the top 10 users by different metrics as a JPG file.
    """
    plt.figure(figsize=(16, 12))
    
    metrics = ['session_frequency', 'total_duration', 'total_dl_traffic', 'total_ul_traffic']
    titles = [
        'Top 10 Users by Session Frequency',
        'Top 10 Users by Total Duration',
        'Top 10 Users by Total DL Traffic',
        'Top 10 Users by Total UL Traffic'
    ]

    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        plt.bar(top_users_dict[metric]['IMSI'].astype(str), top_users_dict[metric][metric], color='skyblue')
        plt.title(titles[i])
        plt.xlabel('User IMSI')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_users_grid.jpg'), format='jpg')
    plt.close()

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
    
    # Ensure the output directories exist
    if not os.path.exists('engagement_and_apps_results'):
        os.makedirs('engagement_and_apps_results')
    if not os.path.exists('clustering_results'):
        os.makedirs('clustering_results')
    if not os.path.exists('pca_results'):
        os.makedirs('pca_results')
    
    # Identify engaged users and most used applications
    identify_engaged_users_and_apps(df)
    
    # Perform K-Means clustering
    normalize_and_cluster(df)
    
    # Perform PCA analysis
    pca_analysis(df)
