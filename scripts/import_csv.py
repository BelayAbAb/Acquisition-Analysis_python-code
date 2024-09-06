import pandas as pd
import psycopg2

# Path to your CSV file
csv_file_path = r'C:\Users\User\Desktop\10Acadamy\Week-2\data\Week1_challenge_data_source.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Print DataFrame info for debugging
print("DataFrame Info:")
print(df.info())
print("DataFrame Columns:")
print(df.columns.tolist())

# Database connection parameters
DATABASE = 'telecom'
USER = 'postgres'
PASSWORD = 'belay'
HOST = 'localhost'
PORT = 5432

# Default PostgreSQL port is 5432


# Table name to import data into
TABLE_NAME = 'public.xdr_data'

# Define the columns and placeholders
columns = [
   "Bearer Id", "Start", "Start ms", "End", "End ms", "Dur. (ms)", "IMSI", "MSISDN/Number", "IMEI",
    "Last Location Name", "Avg RTT DL (ms)", "Avg RTT UL (ms)", "Avg Bearer TP DL (kbps)", "Avg Bearer TP UL (kbps)",
    "TCP DL Retrans. Vol (Bytes)", "TCP UL Retrans. Vol (Bytes)", "DL TP < 50 Kbps (%)", "50 Kbps < DL TP < 250 Kbps (%)",
    "250 Kbps < DL TP < 1 Mbps (%)", "DL TP > 1 Mbps (%)", "UL TP < 10 Kbps (%)", "10 Kbps < UL TP < 50 Kbps (%)",
    "50 Kbps < UL TP < 300 Kbps (%)", "UL TP > 300 Kbps (%)", "HTTP DL (Bytes)", "HTTP UL (Bytes)",
    "Activity Duration DL (ms)", "Activity Duration UL (ms)", "Dur. (ms).1", "Handset Manufacturer",
    "Handset Type", "Nb of sec with 125000B < Vol DL", "Nb of sec with 1250B < Vol UL < 6250B",
    "Nb of sec with 31250B < Vol DL < 125000B", "Nb of sec with 37500B < Vol UL", "Nb of sec with 6250B < Vol DL < 31250B",
    "Nb of sec with 6250B < Vol UL < 37500B", "Nb of sec with Vol DL < 6250B", "Nb of sec with Vol UL < 1250B",
    "Social Media DL (Bytes)", "Social Media UL (Bytes)", "Google DL (Bytes)", "Google UL (Bytes)", "Email DL (Bytes)",
    "Email UL (Bytes)", "Youtube DL (Bytes)", "Youtube UL (Bytes)", "Netflix DL (Bytes)", "Netflix UL (Bytes)",
    "Gaming DL (Bytes)", "Gaming UL (Bytes)", "Other DL (Bytes)", "Other UL (Bytes)", "Total UL (Bytes)", "Total DL (Bytes)"
]

# Create the SQL insert query
placeholders = ', '.join(['%s'] * len(columns))
columns_string = ', '.join(columns)
insert_query = f"INSERT INTO {TABLE_NAME} ({columns_string}) VALUES ({placeholders})"

# Establish a connection to the PostgreSQL database
conn = psycopg2.connect(
    dbname=DATABASE,
    user=USER,
    password=PASSWORD,
    host=HOST,
    port=PORT
)

# Create a cursor object
cur = conn.cursor()

# Debugging: Print a sample of the data and the expected number of columns
print("Number of columns in DataFrame:", len(df.columns))
print("Sample of DataFrame rows:")
for i, row in df.head(5).iterrows():
    print(f"Row {i} length: {len(row)}")
    print(f"Row {i} data: {row.tolist()}")

# Iterate over the rows of the DataFrame and insert data
for index, row in df.iterrows():
    try:
        # Convert 'undefined' and NaN to None, and ensure numerical conversion
        row_values = []
        for col, val in zip(columns, row):
            if pd.isna(val) or val == 'undefined':
                row_values.append(None)
            else:
                # Try to convert to float for numeric fields
                try:
                    row_values.append(float(val))
                except ValueError:
                    row_values.append(val)
       
        # Debugging: Print the length of row_values to check consistency
        if len(row_values) != len(columns):
            print(f"Warning: Row {index} length {len(row_values)} does not match columns length {len(columns)}")
            continue
       
        cur.execute(insert_query, tuple(row_values))
       
    except Exception as e:
        print(f"Error inserting row {index}: {e}")
        print(f"Row data: {row_values}")

# Commit the transaction
conn.commit()

# Close the cursor and connection
cur.close()
conn.close()

print("Data imported successfully!")