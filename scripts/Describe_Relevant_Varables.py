import os
from PIL import Image, ImageDraw, ImageFont

# Define the field descriptions with data types
field_descriptions = {
    "bearer id": "xDR session identifier (double precision)",
    "Start": "Start time of the xDR (text)",
    "Start ms": "Milliseconds offset of start time for the xDR (double precision)",
    "End": "End time of the xDR (text)",
    "End ms": "Milliseconds offset of end time of the xDR (double precision)",
    "Dur. (ms)": "Total Duration of the xDR (in ms) (double precision)",
    "IMSI": "International Mobile Subscriber Identity (double precision)",
    "MSISDN/Number": "MS International PSTN/ISDN Number of mobile - customer number (double precision)",
    "IMEI": "International Mobile Equipment Identity (double precision)",
    "Last Location Name": "User location call name (2G/3G/4G) at the end of the bearer (text)",
    "Avg RTT DL (ms)": "Average Round Trip Time measurement Downlink direction (milliseconds) (double precision)",
    "Avg RTT UL (ms)": "Average Round Trip Time measurement Uplink direction (milliseconds) (double precision)",
    "Avg Bearer TP DL (kbps)": "Average Bearer Throughput for Downlink (kbps) - based on BDR duration (double precision)",
    "Avg Bearer TP UL (kbps)": "Average Bearer Throughput for uplink (kbps) - based on BDR duration (double precision)",
    "TCP DL Retrans. Vol (Bytes)": "TCP volume of Downlink packets detected as retransmitted (bytes) (double precision)",
    "TCP UL Retrans. Vol (Bytes)": "TCP volume of Uplink packets detected as retransmitted (bytes) (double precision)",
    "DL TP < 50 Kbps (%)": "Duration ratio when Bearer Downlink Throughput < 50 Kbps (double precision)",
    "50 Kbps < DL TP < 250 Kbps (%)": "Duration ratio when Bearer Downlink Throughput range is 50 Kbps to 250 Kbps (double precision)",
    "250 Kbps < DL TP < 1 Mbps (%)": "Duration ratio when Bearer Downlink Throughput range is 250 Kbps to 1 Mbps (double precision)",
    "DL TP > 1 Mbps (%)": "Duration ratio when Bearer Downlink Throughput > 1 Mbps (double precision)",
    "UL TP < 10 Kbps (%)": "Duration ratio when Bearer Uplink Throughput < 10 Kbps (double precision)",
    "10 Kbps < UL TP < 50 Kbps (%)": "Duration ratio when Bearer Uplink Throughput range is 10 Kbps to 50 Kbps (double precision)",
    "50 Kbps < UL TP < 300 Kbps (%)": "Duration ratio when Bearer Uplink Throughput range is 50 Kbps to 300 Kbps (double precision)",
    "UL TP > 300 Kbps (%)": "Duration ratio when Bearer Uplink Throughput > 300 Kbps (double precision)",
    "HTTP DL (Bytes)": "HTTP data volume (in Bytes) received by the MS during this session (double precision)",
    "HTTP UL (Bytes)": "HTTP data volume (in Bytes) sent by the MS during this session (double precision)",
    "Activity Duration DL (ms)": "Activity Duration for downlink (ms) - excluding periods of inactivity > 500 ms (double precision)",
    "Activity Duration UL (ms)": "Activity Duration for uplink (ms) - excluding periods of inactivity > 500 ms (double precision)",
    "Dur. (ms).1": "Total Duration of the xDR (in ms) (double precision)",
    "Handset Manufacturer": "Handset manufacturer (text)",
    "Handset Type": "Handset type of the mobile device (text)",
    "Nb of sec with 125000B < Vol DL": "Number of seconds with IP Volume DL > 125000B (double precision)",
    "Nb of sec with 1250B < Vol UL < 6250B": "Number of seconds with IP Volume UL between 1250B and 6250B (double precision)",
    "Nb of sec with 31250B < Vol DL < 125000B": "Number of seconds with IP Volume DL between 31250B and 125000B (double precision)",
    "Nb of sec with 37500B < Vol UL": "Number of seconds with IP Volume UL > 37500B (double precision)",
    "Nb of sec with 6250B < Vol DL < 31250B": "Number of seconds with IP Volume DL between 6250B and 31250B (double precision)",
    "Nb of sec with 6250B < Vol UL < 37500B": "Number of seconds with IP Volume UL between 6250B and 37500B (double precision)",
    "Nb of sec with Vol DL < 6250B": "Number of seconds with IP Volume DL < 6250B (double precision)",
    "Nb of sec with Vol UL < 1250B": "Number of seconds with IP Volume UL < 1250B (double precision)",
    "Social Media DL (Bytes)": "Social Media data volume (in Bytes) received by the MS during this session (double precision)",
    "Social Media UL (Bytes)": "Social Media data volume (in Bytes) sent by the MS during this session (double precision)",
    "Google DL (Bytes)": "Google data volume (in Bytes) received by the MS during this session (double precision)",
    "Google UL (Bytes)": "Google data volume (in Bytes) sent by the MS during this session (double precision)",
    "Email DL (Bytes)": "Email data volume (in Bytes) received by the MS during this session (double precision)",
    "Email UL (Bytes)": "Email data volume (in Bytes) sent by the MS during this session (double precision)",
    "Youtube DL (Bytes)": "YouTube data volume (in Bytes) received by the MS during this session (double precision)",
    "Youtube UL (Bytes)": "YouTube data volume (in Bytes) sent by the MS during this session (double precision)",
    "Netflix DL (Bytes)": "Netflix data volume (in Bytes) received by the MS during this session (double precision)",
    "Netflix UL (Bytes)": "Netflix data volume (in Bytes) sent by the MS during this session (double precision)",
    "Gaming DL (Bytes)": "Gaming data volume (in Bytes) received by the MS during this session (double precision)",
    "Gaming UL (Bytes)": "Gaming data volume (in Bytes) sent by the MS during this session (double precision)",
    "Other DL (Bytes)": "Other data volume (in Bytes) received by the MS during this session (double precision)",
    "Other UL (Bytes)": "Other data volume (in Bytes) sent by the MS during this session (double precision)",
    "Total DL (Bytes)": "Data volume (in Bytes) received by the MS during this session (IP layer + overhead) (double precision)",
    "Total UL (Bytes)": "Data volume (in Bytes) sent by the MS during this session (IP layer + overhead) (double precision)"
}

def generate_description_image():
    """
    Generates an image with descriptions of fields and their meanings, arranged in three columns, and saves it as a .jpg file.
    """
    # Create an image with PIL
    width, height = 3000, 2000  # Increased width for three columns
    background_color = (255, 255, 255)  # White background
    text_color = (0, 0, 0)  # Black text
    font_size = 16
    column_width = width // 3 - 20  # Space for three columns with padding

    # Create a blank image
    image = Image.new('RGB', (width, height), background_color)
    draw = ImageDraw.Draw(image)
   
    # Load a font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
   
    # Calculate column positions
    column1_x = 10
    column2_x = column1_x + column_width + 10
    column3_x = column2_x + column_width + 10
    y_text = 10
    column_switch_threshold = height // 3  # Switch column if exceeded this height

    # Initial column to write to
    column_x = column1_x

    # Loop through the descriptions
    for i, (field, description) in enumerate(field_descriptions.items()):
        text = f"{field}: {description}"
        bbox = draw.textbbox((column_x, y_text), text, font=font)
        text_height = bbox[3] - bbox[1]  # Height of the bounding box

        if y_text + text_height > column_switch_threshold:
            if column_x == column1_x:
                # Switch to second column
                y_text = 10
                column_x = column2_x
            elif column_x == column2_x:
                # Switch to third column
                y_text = 10
                column_x = column3_x
            else:
                # Reset to first column if all columns are used
                y_text = 10
                column_x = column1_x
       
        draw.text((column_x, y_text), text, font=font, fill=text_color)
        y_text += text_height + 5  # Add some padding between lines

    # Save the image
    output_filename = "field_descriptions_three_columns.jpg"
    image.save(output_filename)
    print(f"Image saved as {output_filename}")

# Generate the image
generate_description_image()
