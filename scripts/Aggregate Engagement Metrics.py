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

def aggregate_engagement_metrics(df):
    """
    Compute engagement metrics such as session frequency, duration, and traffic for each user.
    """
    if df is not None:
        # Ensure the necessary columns exist
        required_columns = ['IMSI', 'MSISDN/Number', 'Dur. (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Missing columns for aggregation: {missing_columns}")
            return

        # Aggregate metrics
        metrics_df = df.groupby('IMSI').agg(
            session_frequency=pd.NamedAgg(column='IMSI', aggfunc='count'),
            total_duration=pd.NamedAgg(column='Dur. (ms)', aggfunc='sum'),
            total_dl_traffic=pd.NamedAgg(column='Total DL (Bytes)', aggfunc='sum'),
            total_ul_traffic=pd.NamedAgg(column='Total UL (Bytes)', aggfunc='sum')
        ).reset_index()

        # Identify top 10 users for each metric
        top_users = {}
        for metric in ['session_frequency', 'total_duration', 'total_dl_traffic', 'total_ul_traffic']:
            top_users[metric] = metrics_df.nlargest(10, metric)
        
        # Ensure the output directory exists
        output_dir = 'engagement_metrics_results'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save the grid of top users plots
        save_top_users_grid(top_users, output_dir)
        
        # Print the top 10 users for each metric
        for metric, df in top_users.items():
            print(f"\nTop 10 Users by {metric.replace('_', ' ').title()}:")
            print(df[['IMSI', metric]])
    else:
        print("Failed to retrieve data.")

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
    
    # Ensure the output directories exist
    if not os.path.exists('engagement_metrics_results'):
        os.makedirs('engagement_metrics_results')
    
    # Aggregate engagement metrics
    aggregate_engagement_metrics(df)
    
    # Perform PCA analysis
    pca_analysis(df)
