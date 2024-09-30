import requests
import pandas as pd
import io
import time  # For handling API rate limits
import os

# Your AlphaVantage API key
with open('/Users/perrylin/quan_fin/fx_api_key.txt', 'r') as file:
    api_key = file.read().strip()


# List of fiat currencies
currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']


# Generate currency pairs (excluding pairs where base and quote are the same)
currency_pairs = []
for base in currencies:
    for quote in currencies:
        if base != quote:
            currency_pairs.append((base, quote))

# Directory to save fetched data
data_dir = 'currency_data'
os.makedirs(data_dir, exist_ok=True)


# Function to fetch hourly data for a currency pair and save it
def fetch_and_save_hourly_fx_data(from_symbol, to_symbol):
    filename = f"{data_dir}/{from_symbol}_{to_symbol}_hourly.csv"
    if os.path.exists(filename):
        print(f"Data for {from_symbol}/{to_symbol} already exists. Skipping download.")
    else:
        print(f"Fetching data for {from_symbol}/{to_symbol}...")
        url = (
            f'https://www.alphavantage.co/query?function=FX_INTRADAY'
            f'&from_symbol={from_symbol}&to_symbol={to_symbol}'
            f'&interval=60min&outputsize=full&apikey={api_key}&datatype=csv'
        )
        response = requests.get(url)
        if response.status_code == 200:
            # Read the CSV data into a DataFrame
            data = pd.read_csv(io.StringIO(response.text))
            # Save the data to a CSV file
            data.to_csv(filename, index=False)
            print(f"Data for {from_symbol}/{to_symbol} saved to {filename}.")
        else:
            print(f"Error fetching data for {from_symbol}/{to_symbol}: {response.status_code}")
            # Optionally, handle the error or retry


# Fetch and save data for all currency pairs
for from_symbol, to_symbol in currency_pairs:
    fetch_and_save_hourly_fx_data(from_symbol, to_symbol)
    # Pause to respect API rate limits
    time.sleep(12)  # AlphaVantage recommends up to 5 API calls per minute
