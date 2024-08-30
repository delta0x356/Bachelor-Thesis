import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.regression.linear_model import GLS
import os
from datetime import datetime, timedelta

def prepare_data(file_path, metric_name, mc_data, fgi_data):
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
    df = pd.merge(df, extended_mc_data, on='Date', how='left')
    df = pd.merge(df, extended_fgi_data[['Date', 'Fear_Greed_Index']], on='Date', how='left')

    print(f"Protocol: {file_path}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Airdrop date: {airdrop_date}")
    print(f"Market cap data available: {df['MCt'].notna().sum()} / {len(df)} days")
    print(f"Fear and Greed Index data available: {df['Fear_Greed_Index'].notna().sum()} / {len(df)} days")
    print("First few rows of merged data:")
    print(df.head())
    print("\n")

    return df, airdrop_date

def fit_model(df, metric_name):
    X = sm.add_constant(df[['T', 'X', 'X_T', 't', 't_X', 'MCt', 'Fear_Greed_Index']])
    y = df[metric_name]

    ols_model = sm.OLS(y, X).fit()

    _, p_value, _, _ = het_breuschpagan(ols_model.resid, ols_model.model.exog)

    if p_value < 0.05:
        print(f"Heteroskedasticity detected for {metric_name} (p-value: {p_value:.4f}). Using GLS.")
        weights = 1 / (ols_model.resid ** 2)
        gls_model = GLS(y, X, weights=weights).fit()
        return gls_model
    else:
        print(f"No significant heteroskedasticity detected for {metric_name} (p-value: {p_value:.4f}). Using OLS.")
        return ols_model

def analyze_protocol_type(folder_path, metric_name, protocol_type, mc_data, fgi_data):
    all_results = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            df, airdrop_date = prepare_data(file_path, metric_name, mc_data, fgi_data)
            results = fit_model(df, metric_name)

            all_results.append({
                'protocol': file.split('.')[0],
                'α0 or β0 or γ0 (Intercept)': results.params['const'],
                'α1 or β1 or γ1 (T)': results.params['T'],
                'α2 or β2 or γ2 (X)': results.params['X'],
                'α3 or β3 or γ3 (X*T)': results.params['X_T'],
                'α4 or β4 or γ4 (t)': results.params['t'],
                'α5 or β5 or γ5 (t*X)': results.params['t_X'],
                'δ1 (MCt)': results.params['MCt'],
                'δ2 (FGI)': results.params['Fear_Greed_Index'],
                'p_value (X)': results.pvalues['X'],
                'p_value (X*T)': results.pvalues['X_T'],
                'p_value (MCt)': results.pvalues['MCt'],
                'p_value (FGI)': results.pvalues['Fear_Greed_Index'],
                'airdrop_date': airdrop_date
            })

    return pd.DataFrame(all_results)

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

for folder, metric, protocol_type in protocol_types:
    print(f"\nAnalysis for {protocol_type}:")
    results = analyze_protocol_type(folder, metric, protocol_type, mc_data, fgi_data)

    if results.empty:
        print(f"No results for {protocol_type}. Skipping...")
        continue

    print(results)
    print(f"\n{protocol_type} Average Effects:")
    print(f"Average immediate effect: {results['α2 or β2 or γ2 (X)'].mean()}")
    print(f"Average slope change: {results['α3 or β3 or γ3 (X*T)'].mean()}")
    print(f"Average market cap effect: {results['δ1 (MCt)'].mean()}")
    print(f"Average Fear and Greed Index effect: {results['δ2 (FGI)'].mean()}")
    print(f"Protocols with significant immediate effect: {(results['p_value (X)'] < 0.05).sum()}/{len(results)}")
    print(f"Protocols with significant slope change: {(results['p_value (X*T)'] < 0.05).sum()}/{len(results)}")
    print(f"Protocols with significant market cap effect: {(results['p_value (MCt)'] < 0.05).sum()}/{len(results)}")
    print(f"Protocols with significant Fear and Greed Index effect: {(results['p_value (FGI)'] < 0.05).sum()}/{len(results)}")

    result_folder = 'result'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    results.to_csv(os.path.join(result_folder, f'{protocol_type.lower()}_analysis_results.csv'), index=False)

print(f"\nResults have been saved in the '{result_folder}' folder.")