import requests
import pandas as pd
from datetime import datetime

# Make the API request
url = "https://api.alternative.me/fng/?limit=605&date_format=world&start=1672531200&end=1724889600"
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()['data']
    
    # Create a list to store the processed data
    processed_data = []
    
    # Process each data point
    for item in data:
        # Check if the timestamp is already a date string
        if '-' in item['timestamp']:
            date = datetime.strptime(item['timestamp'], '%d-%m-%Y').strftime('%Y-%m-%d')
        else:
            # If it's a Unix timestamp, convert it
            date = datetime.fromtimestamp(int(item['timestamp'])).strftime('%Y-%m-%d')
        
        processed_data.append({
            'Date': date,
            'Fear_Greed_Index': int(item['value']),
            'Classification': item['value_classification']
        })
    
    # Create a pandas DataFrame
    df = pd.DataFrame(processed_data)
    
    # Set the Date column as the index
    df.set_index('Date', inplace=True)
    
    # Sort the DataFrame by date
    df.sort_index(inplace=True)
    
    # Display the first few rows of the DataFrame
    print(df.head())
    
    # Save the DataFrame to a CSV file
    df.to_csv('fear_greed_index.csv')
    print("Data saved to fear_greed_index.csv")
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")