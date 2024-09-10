import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

# Define your connection parameters
DB_PARAMS = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "belay",
    "host": "localhost",
    "port": "5432"
}

# Create SQLAlchemy engine
DATABASE_URL = f"postgresql+psycopg2://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}"
engine = create_engine(DATABASE_URL)

# Load data from PostgreSQL
@st.cache_data
def load_data():
    try:
        query = "SELECT * FROM xdr_data"
        df = pd.read_sql(query, engine)
        return df
    except SQLAlchemyError as e:
        st.error(f"An error occurred while connecting to the database: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

df = load_data()

# Set page title
st.title("Telecom Data - Exploratory Data Analysis")

# Show first few rows of the dataset
st.subheader("Initial Data")
if not df.empty:
    st.write(df.head())
else:
    st.write("No data available.")

# Option to drop or fill missing values
st.subheader("Handle Missing Values")
if not df.empty:
    missing_columns = df.columns[df.isnull().any()]
    if missing_columns.size > 0:
        missing_column = st.selectbox("Choose a column to fill missing values", missing_columns)
        fill_method = st.radio("Fill method", ["Fill with Mean", "Fill with Median", "Drop Rows"])

        if st.button("Apply Fill"):
            if fill_method == "Fill with Mean":
                df[missing_column] = df[missing_column].fillna(df[missing_column].mean())
            elif fill_method == "Fill with Median":
                df[missing_column] = df[missing_column].fillna(df[missing_column].median())
            else:
                df = df.dropna(subset=[missing_column])
            st.success(f"{missing_column} cleaned successfully")
    else:
        st.write("No missing values to handle.")

# ---- 2. Descriptive Statistics ----
st.subheader("Descriptive Statistics")
if not df.empty:
    st.write(df.describe())
else:
    st.write("No data available for descriptive statistics.")

# ---- 3. Visualizations ----

# Handset Type Distribution
st.subheader("Handset Type Distribution")
if 'Handset Type' in df.columns:
    handset_counts = df['Handset Type'].value_counts()
    st.bar_chart(handset_counts)
else:
    st.write("Column 'Handset Type' not found in the dataset.")

# Comparing Handset Type with another feature, e.g., Total DL (Bytes)
st.subheader("Comparison: Handset Type vs. Total DL (Bytes)")
if 'Handset Type' in df.columns and 'Total DL (Bytes)' in df.columns:
    df_clean = df.dropna(subset=["Handset Type", "Total DL (Bytes)"])  # Clean data

    fig, ax = plt.subplots()
    sns.boxplot(x="Handset Type", y="Total DL (Bytes)", data=df_clean, ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)
else:
    st.write("Required columns not found in the dataset.")