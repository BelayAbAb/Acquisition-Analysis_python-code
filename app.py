import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv
import psycopg2
import datetime

# Load environment variables from .env file
load_dotenv()

# Define paths for performance tracking
PERFORMANCE_LOG_PATH = 'model_performance_log.csv'

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

def prepare_data(df):
    """
    Prepare the data for regression modeling.
    """
    if df is not None:
        df.fillna(0, inplace=True)
        df['Start'] = pd.to_datetime(df['Start'], errors='coerce')
        df['End'] = pd.to_datetime(df['End'], errors='coerce')
       
        features = ['Dur. (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)']
        target = 'Overall Satisfaction'
       
        if target not in df.columns:
            df[target] = (df['Total DL (Bytes)'] + df['Total UL (Bytes)']) / 2
       
        X = df[features]
        y = df[target]
       
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
       
        return X_scaled, y
    else:
        st.error("Failed to retrieve data.")
        return None, None

def build_and_evaluate_regression_model(X, y):
    """
    Build and evaluate a regression model to predict satisfaction scores.
    """
    if X is not None and y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
       
        model = LinearRegression()
        model.fit(X_train, y_train)
       
        y_pred = model.predict(X_test)
       
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
       
        # Log performance metrics
        log_performance(mae, mse, r2)
       
        st.write("### Regression Model Evaluation")
        st.write(f"Mean Absolute Error: {mae}")
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R-squared (R^2): {r2}")
       
        results_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        })
        st.write("### Model Predictions")
        st.dataframe(results_df.head())
    else:
        st.error("Failed to prepare data for modeling.")

def clean_data(df):
    """
    Clean the data by handling missing values and incorrect data types.
    """
    if df is not None:
        st.write("### Initial Data Info")
        st.write(df.info())
        st.write(df.describe())
       
        st.write("### Missing Values")
        st.write(df.isnull().sum())
       
        fill_value = st.selectbox("Select a fill value for missing data", [0, 'mean', 'median'])
        if fill_value == 'mean':
            df.fillna(df.mean(), inplace=True)
        elif fill_value == 'median':
            df.fillna(df.median(), inplace=True)
        else:
            df.fillna(0, inplace=True)
       
        st.write("### Data After Cleaning")
        st.write(df.info())
        st.write(df.describe())
       
        return df
    else:
        st.error("Failed to retrieve data.")
        return None

def log_performance(mae, mse, r2):
    """
    Log model performance metrics to a CSV file.
    """
    log_entry = {
        'Timestamp': datetime.datetime.now(),
        'Mean Absolute Error': mae,
        'Mean Squared Error': mse,
        'R-squared': r2
    }
   
    log_df = pd.DataFrame([log_entry])
   
    if os.path.exists(PERFORMANCE_LOG_PATH):
        existing_df = pd.read_csv(PERFORMANCE_LOG_PATH)
        log_df = pd.concat([existing_df, log_df], ignore_index=True)
   
    log_df.to_csv(PERFORMANCE_LOG_PATH, index=False)

def visualize_performance():
    """
    Visualize historical model performance metrics.
    """
    if os.path.exists(PERFORMANCE_LOG_PATH):
        log_df = pd.read_csv(PERFORMANCE_LOG_PATH)
        st.write("### Model Performance Over Time")
       
        fig, ax = plt.subplots(3, 1, figsize=(10, 15))

        ax[0].plot(pd.to_datetime(log_df['Timestamp']), log_df['Mean Absolute Error'], marker='o', label='Mean Absolute Error')
        ax[0].set_title('Mean Absolute Error Over Time')
        ax[0].set_xlabel('Timestamp')
        ax[0].set_ylabel('MAE')
        ax[0].legend()

        ax[1].plot(pd.to_datetime(log_df['Timestamp']), log_df['Mean Squared Error'], marker='o', label='Mean Squared Error', color='orange')
        ax[1].set_title('Mean Squared Error Over Time')
        ax[1].set_xlabel('Timestamp')
        ax[1].set_ylabel('MSE')
        ax[1].legend()

        ax[2].plot(pd.to_datetime(log_df['Timestamp']), log_df['R-squared'], marker='o', label='R-squared', color='green')
        ax[2].set_title('R-squared Over Time')
        ax[2].set_xlabel('Timestamp')
        ax[2].set_ylabel('R^2')
        ax[2].legend()

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No performance data found.")

def main():
    st.title("Regression Model for Predicting Satisfaction Scores")

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

    df = load_data(query)

    tab1, tab2, tab3, tab4 = st.tabs(["Data Cleaning", "Prepare Data", "Regression Model", "Performance Monitoring"])

    with tab1:
        st.header("Data Cleaning")
        df_cleaned = clean_data(df)
        st.write(df_cleaned.head())

    with tab2:
        st.header("Prepare Data")
        X, y = prepare_data(df_cleaned)
        if X is not None and y is not None:
            st.write("Feature matrix (X) and target vector (y) are prepared.")
            st.write("Feature matrix (X):")
            st.write(X[:5])
            st.write("Target vector (y):")
            st.write(y.head())

    with tab3:
        st.header("Build and Evaluate Regression Model")
        if X is not None and y is not None:
            build_and_evaluate_regression_model(X, y)
        else:
            st.error("Data is not prepared correctly. Please ensure data cleaning and preparation steps are done.")

    with tab4:
        st.header("Model Performance Monitoring")
        visualize_performance()

if __name__ == "__main__":
    main()

