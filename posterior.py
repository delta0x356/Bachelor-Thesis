import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
import os
from datetime import datetime, timedelta

def prepare_data(file_path, metric_name, mc_data, fgi_data):
    try:
        print(f"Processing file: {file_path}")
        df = pd.read_csv(file_path)
        print(f"Columns in df: {df.columns}")
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df = df.sort_values('Date')

        airdrop_date = df.iloc[30]['Date']

        df['T'] = range(-30, 31)
        df['X'] = np.where(df.index < 30, 0, 1)
        df['X_T'] = df['X'] * df['T']
        df['t'] = range(1, 62)
        df['t_X'] = df['t'] * df['X']

        start_date = df['Date'].min() - timedelta(days=1)
        end_date = df['Date'].max() + timedelta(days=1)

        # Prepare market cap data
        extended_mc_data = mc_data[(mc_data['Date'] >= start_date) & (mc_data['Date'] <= end_date)]
        date_range = pd.date_range(start=start_date, end=end_date)
        extended_mc_data = extended_mc_data.set_index('Date').reindex(date_range).reset_index()
        extended_mc_data = extended_mc_data.rename(columns={'index': 'Date'})
        extended_mc_data['MCt'] = extended_mc_data['MCt'].ffill().bfill()

        # Prepare Fear and Greed Index data
        extended_fgi_data = fgi_data[(fgi_data['Date'] >= start_date) & (fgi_data['Date'] <= end_date)]
        extended_fgi_data = extended_fgi_data.set_index('Date').reindex(date_range).reset_index()
        extended_fgi_data = extended_fgi_data.rename(columns={'index': 'Date'})
        extended_fgi_data['Fear_Greed_Index'] = extended_fgi_data['Fear_Greed_Index'].ffill().bfill()

        # Merge all data
        df = pd.merge(df, extended_mc_data[['Date', 'MCt']], on='Date', how='left')
        df = pd.merge(df, extended_fgi_data[['Date', 'Fear_Greed_Index']], on='Date', how='left')

        print(f"Final columns in df: {df.columns}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Airdrop date: {airdrop_date}")
        print(f"Market cap data available: {df['MCt'].notna().sum()} / {len(df)} days")
        print(f"Fear and Greed Index data available: {df['Fear_Greed_Index'].notna().sum()} / {len(df)} days")
        print("First few rows of merged data:")
        print(df.head())
        print("\n")

        return df
    except Exception as e:
        print(f"Error in prepare_data for {file_path}: {str(e)}")
        raise

def fit_model(df, metric_name):
    try:
        X = sm.add_constant(df[['T', 'X', 'X_T', 't', 't_X', 'MCt', 'Fear_Greed_Index']])
        y = df[metric_name]
        model = sm.OLS(y, X).fit()
        return model
    except Exception as e:
        print(f"Error in fit_model for {metric_name}: {str(e)}")
        raise

def check_autocorrelation(model):
    return durbin_watson(model.resid)

def analyze_protocol_type(folder_path, metric_name, protocol_type, mc_data, fgi_data):
    autocorrelation_results = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            try:
                file_path = os.path.join(folder_path, file)
                df = prepare_data(file_path, metric_name, mc_data, fgi_data)
                model = fit_model(df, metric_name)
                dw_statistic = check_autocorrelation(model)

                autocorrelation_results.append({
                    'protocol': file.split('.')[0],
                    'Durbin-Watson statistic': dw_statistic,
                    'Autocorrelation': 'Positive' if dw_statistic < 1.5 else ('Negative' if dw_statistic > 2.5 else 'No evidence')
                })
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")

    return pd.DataFrame(autocorrelation_results)

# Load market capitalization data
mc_data = pd.read_csv('market_cap.csv')
mc_data['Date'] = pd.to_datetime(mc_data['Date'], format='%d/%m/%Y')
print("Market cap data loaded. Date range:", mc_data['Date'].min(), "to", mc_data['Date'].max())

# Load Fear and Greed Index data
fgi_data = pd.read_csv('fear_greed_index.csv')
fgi_data['Date'] = pd.to_datetime(fgi_data['Date'], format='%d/%m/%Y')
print("Fear and Greed Index data loaded. Date range:", fgi_data['Date'].min(), "to", fgi_data['Date'].max())

# Analyze each protocol type
protocol_types = [('TVL', 'TVL', 'DeFi'), ('volume', 'Volume', 'DEX'), ('DAU', 'DAU', 'SocialFi')]

# Create posterior_analysis folder if it doesn't exist
posterior_folder = 'posterior_analysis'
if not os.path.exists(posterior_folder):
    os.makedirs(posterior_folder)

for folder, metric, protocol_type in protocol_types:
    print(f"\nAutocorrelation analysis for {protocol_type}:")
    results = analyze_protocol_type(folder, metric, protocol_type, mc_data, fgi_data)

    print(results)

    # Save results
    results.to_csv(os.path.join(posterior_folder, f'{protocol_type.lower()}_autocorrelation_results.csv'), index=False)

print(f"\nAutocorrelation analysis results have been saved in the '{posterior_folder}' folder.")