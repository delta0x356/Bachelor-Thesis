import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.regression.linear_model import GLS
import os
from datetime import datetime, timedelta

def prepare_data(file_path, metric_name, mc_data, sp500_data):
    df = pd.read_csv(file_path)
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
    
    # Extend market cap data
    extended_mc_data = mc_data[(mc_data['Date'] >= start_date) & (mc_data['Date'] <= end_date)]
    date_range = pd.date_range(start=start_date, end=end_date)
    extended_mc_data = extended_mc_data.set_index('Date').reindex(date_range).reset_index()
    extended_mc_data = extended_mc_data.rename(columns={'index': 'Date'})
    extended_mc_data['MCt'] = extended_mc_data['MCt'].ffill().bfill()
    
    # Extend S&P 500 data
    extended_sp500_data = sp500_data[(sp500_data['Date'] >= start_date) & (sp500_data['Date'] <= end_date)]
    extended_sp500_data = extended_sp500_data.set_index('Date').reindex(date_range).reset_index()
    extended_sp500_data = extended_sp500_data.rename(columns={'index': 'Date'})
    extended_sp500_data['Close'] = extended_sp500_data['Close'].ffill().bfill()
    
    # Merge all data
    df = pd.merge(df, extended_mc_data, on='Date', how='left')
    df = pd.merge(df, extended_sp500_data, on='Date', how='left')

    print(f"Protocol: {file_path}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Airdrop date: {airdrop_date}")
    print(f"Market cap data available: {df['MCt'].notna().sum()} / {len(df)} days")
    print(f"S&P 500 data available: {df['Close'].notna().sum()} / {len(df)} days")
    print("First few rows of merged data:")
    print(df.head())
    print("\n")

    return df, airdrop_date

def fit_model(df, metric_name):
    X = sm.add_constant(df[['T', 'X', 'X_T', 't', 't_X', 'MCt', 'Close']])
    y = df[metric_name]

    # First, fit OLS model
    ols_model = sm.OLS(y, X).fit()

    # Test for heteroskedasticity
    _, p_value, _, _ = het_breuschpagan(ols_model.resid, ols_model.model.exog)
    
    print(f"Heteroskedasticity test p-value: {p_value}")

    # If heteroskedasticity is detected, use GLS
    if p_value < 0.05:
        print("Heteroskedasticity detected. Using GLS.")
        weights = 1 / (ols_model.resid ** 2)
        gls_model = GLS(y, X, weights=weights).fit()
        return gls_model
    else:
        print("No heteroskedasticity detected. Using OLS.")
        return ols_model

def analyze_protocol_type(folder_path, metric_name, protocol_type, mc_data, sp500_data):
    all_results = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            df, airdrop_date = prepare_data(file_path, metric_name, mc_data, sp500_data)
            results = fit_model(df, metric_name)

            all_results.append({
                'protocol': file.split('.')[0],
                'α0 or β0 or γ0 (Intercept)': results.params['const'],
                'α1 or β1 or γ1 (T)': results.params['T'],
                'α2 or β2 or γ2 (X)': results.params['X'],
                'α3 or β3 or γ3 (X*T)': results.params['X_T'],
                'α4 or β4 or γ4 (t)': results.params['t'],
                'α5 or β5 or γ5 (t*X)': results.params['t_X'],
                'δ (MCt)': results.params['MCt'],
                'S&P 500': results.params['Close'],
                'p_value (X)': results.pvalues['X'],
                'p_value (X*T)': results.pvalues['X_T'],
                'p_value (MCt)': results.pvalues['MCt'],
                'p_value (S&P 500)': results.pvalues['Close'],
                'airdrop_date': airdrop_date
            })

    return pd.DataFrame(all_results)

# Load market capitalization data
mc_data = pd.read_csv('market_cap.csv')
mc_data['Date'] = pd.to_datetime(mc_data['Date'], format='%d/%m/%Y')
print("Market cap data loaded. Date range:", mc_data['Date'].min(), "to", mc_data['Date'].max())

# Load S&P 500 data
sp500_data = pd.read_csv('sp500.csv')
sp500_data['Date'] = pd.to_datetime(sp500_data['Date'], format='%d/%m/%Y')
print("S&P 500 data loaded. Date range:", sp500_data['Date'].min(), "to", sp500_data['Date'].max())

# Analyze each protocol type
protocol_types = [('TVL', 'TVL', 'DeFi'), ('volume', 'Volume', 'DEX'), ('DAU', 'DAU', 'SocialFi')]

for folder, metric, protocol_type in protocol_types:
    print(f"\nAnalysis for {protocol_type}:")
    results = analyze_protocol_type(folder, metric, protocol_type, mc_data, sp500_data)

    if results.empty:
        print(f"No results for {protocol_type}. Skipping...")
        continue

    print(results)
    print(f"\n{protocol_type} Average Effects:")
    print(f"Average immediate effect: {results['α2 or β2 or γ2 (X)'].mean()}")
    print(f"Average slope change: {results['α3 or β3 or γ3 (X*T)'].mean()}")
    print(f"Average market cap effect: {results['δ (MCt)'].mean()}")
    print(f"Average S&P 500 effect: {results['S&P 500'].mean()}")
    print(f"Protocols with significant immediate effect: {(results['p_value (X)'] < 0.05).sum()}/{len(results)}")
    print(f"Protocols with significant slope change: {(results['p_value (X*T)'] < 0.05).sum()}/{len(results)}")
    print(f"Protocols with significant market cap effect: {(results['p_value (MCt)'] < 0.05).sum()}/{len(results)}")
    print(f"Protocols with significant S&P 500 effect: {(results['p_value (S&P 500)'] < 0.05).sum()}/{len(results)}")

    # Create 'result' folder if it doesn't exist
    result_folder = 'result'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Save results to CSV in the 'result' folder
    results.to_csv(os.path.join(result_folder, f'{protocol_type.lower()}_analysis_results.csv'), index=False)

print(f"\nResults have been saved in the '{result_folder}' folder.")