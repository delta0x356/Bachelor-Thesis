import pandas as pd
from datetime import datetime

# Read the CSV file
df = pd.read_csv('sp500_data.csv')

# Function to parse and format the date
def parse_and_format_date(date_string):
    # Parse the date string
    date_obj = datetime.strptime(date_string.split()[0], '%Y-%m-%d')
    # Format the date as dd/mm/yyyy
    return date_obj.strftime('%d/%m/%Y')

# Apply the parsing and formatting function to the 'Date' column
df['Date'] = df['Date'].apply(parse_and_format_date)

# Save the formatted data to a new CSV file
df.to_csv('sp500_data_formatted.csv', index=False)

print("Date formatting completed. Output saved to 'sp500_data_formatted.csv'.")