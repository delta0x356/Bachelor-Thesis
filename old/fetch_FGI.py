import requests
import pandas as pd
from datetime import datetime, timedelta

# Set the start and end dates
start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 8, 27)

# Make the API request
url = f"https://api.alternative.me/fng/?limit=1000&date_format=world&start={int(start_date.timestamp())}&end={int(end_date.timestamp())}"
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()['data']
    
    # Create a dictionary to store the processed data
    processed_data = {}
    
    # Process each data point
    for item in data:
        if '-' in item['timestamp']:
            date = datetime.strptime(item['timestamp'], '%d-%m-%Y')
        else:
            date = datetime.fromtimestamp(int(item['timestamp']))
        
        processed_data[date] = {
            'Fear_Greed_Index': int(item['value']),
            'Classification': item['value_classification']
        }
    
    # Create a date range for all dates
    date_range = pd.date_range(start=start_date, end=end_date)
    
    # Create a list to store the final data
    final_data = []
    
    # Fill in the data for each date
    for date in date_range:
        if date in processed_data:
            final_data.append({
                'Date': date.strftime('%d/%m/%Y'),
                'Fear_Greed_Index': processed_data[date]['Fear_Greed_Index'],
                'Classification': processed_data[date]['Classification']
            })
        else:
            final_data.append({
                'Date': date.strftime('%d/%m/%Y'),
                'Fear_Greed_Index': None,
                'Classification': None
            })
    
    # Create a pandas DataFrame
    df = pd.DataFrame(final_data)
    
    # Set the Date column as the index
    df.set_index('Date', inplace=True)
    
    # Display the first few rows of the DataFrame
    print(df.head())
    
    # Save the DataFrame to a CSV file
    df.to_csv('fear_greed_index.csv')
    print("Data saved to fear_greed_index.csv")
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")