import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import os
from multiprocessing import Pool

# Define directories
data_dir = 'currency_data'
results_dir = 'backtest_results'
plots_dir = os.path.join(results_dir, 'plots')

# Create directories if they don't exist
os.makedirs(plots_dir, exist_ok=True)


def load_and_prepare_data(file_path, rsi_length=2, oversold=20, overbought=90):
    """
    Load CSV data, preprocess it, and calculate RSI.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file.
    rsi_length : int
        Period for RSI calculation.
    oversold : int
        RSI value below which a buy signal is generated.
    overbought : int
        RSI value above which a sell signal is generated.

    Returns:
    --------
    pd.DataFrame
        Preprocessed DataFrame with RSI and Signals.
    """
    # Load the data
    df = pd.read_csv(file_path)

    # Convert 'timestamp' to datetime and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Rename columns to match existing code
    df.rename(columns={
        'timestamp': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
    }, inplace=True)

    # Create 'Adj Close' column
    df['Adj Close'] = df['Close']

    # Calculate RSI
    df['RSI'] = ta.rsi(df['Adj Close'], length=rsi_length)

    # Define buy/sell signals
    df['Signal'] = None
    df.loc[df['RSI'] < oversold, 'Signal'] = 'Buy'    # Oversold
    df.loc[df['RSI'] > overbought, 'Signal'] = 'Sell'   # Overbought

    return df


def backtest_strategy(df, initial_cash=100_000, 
                     risk_percent=0.02, stop_loss_percent=0.02, 
                     take_profit_percent=0.10, transaction_cost=0.0002):
    """
    Backtest the RSI strategy on the provided DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing OHLC data with RSI and Signals.
    initial_cash : float, optional
        Starting cash for the portfolio.
    risk_percent : float, optional
        Percentage of equity to risk per trade.
    stop_loss_percent : float, optional
        Stop-loss percentage below the entry price.
    take_profit_percent : float, optional
        Take-profit percentage above the entry price.
    transaction_cost : float, optional
        Transaction cost per trade as a fraction (e.g., 0.0002 for 0.02%).

    Returns:
    --------
    dict
        Performance metrics.
    pd.DataFrame
        DataFrame with equity curve and Buy-and-Hold equity.
    """
    cash = initial_cash
    units = 0
    equity = initial_cash
    equity_curve = []
    total_trades = 0
    started = False
    purchase_price = 0
    time_in_market_hours = 0

    buy_signals = df[df['Signal'] == 'Buy'].index
    sell_signals = df[df['Signal'] == 'Sell'].index

    for i, row in df.iterrows():
        if not started and i in buy_signals:
            # Calculate position size based on risk
            position_risk = equity * risk_percent
            stop_loss = row['Adj Close'] * (1 - stop_loss_percent)
            risk_per_unit = row['Adj Close'] - stop_loss
            if risk_per_unit <= 0:
                # Prevent division by zero or negative risk
                units_to_buy = 0
            else:
                units_to_buy = int(position_risk // risk_per_unit)

            if units_to_buy > 0 and cash >= units_to_buy * row['Adj Close']:
                units += units_to_buy
                cash -= units_to_buy * row['Adj Close']
                # Subtract transaction cost
                cost = units_to_buy * row['Adj Close'] * transaction_cost
                cash -= cost
                total_trades += 1
                purchase_price = row['Adj Close']
                started = True

        elif started:
            if i in buy_signals and units == 0:
                # Recalculate position size
                position_risk = equity * risk_percent
                stop_loss = row['Adj Close'] * (1 - stop_loss_percent)
                risk_per_unit = row['Adj Close'] - stop_loss
                if risk_per_unit <= 0:
                    units_to_buy = 0
                else:
                    units_to_buy = int(position_risk // risk_per_unit)

                if units_to_buy > 0 and cash >= units_to_buy * row['Adj Close']:
                    units += units_to_buy
                    cash -= units_to_buy * row['Adj Close']
                    # Subtract transaction cost
                    cost = units_to_buy * row['Adj Close'] * transaction_cost
                    cash -= cost
                    total_trades += 1
                    purchase_price = row['Adj Close']

            elif i in sell_signals and units > 0:
                # Sell all units
                proceeds = units * row['Adj Close']
                cash += proceeds
                # Subtract transaction cost
                cost = proceeds * transaction_cost
                cash -= cost
                units = 0
                total_trades += 1

            elif units > 0:
                # Take profit
                if row['Adj Close'] >= purchase_price * (1 + take_profit_percent):
                    proceeds = units * row['Adj Close']
                    cash += proceeds
                    # Subtract transaction cost
                    cost = proceeds * transaction_cost
                    cash -= cost
                    units = 0
                    total_trades += 1
                # Stop-loss
                elif row['Adj Close'] <= purchase_price * (1 - stop_loss_percent):
                    proceeds = units * row['Adj Close']
                    cash += proceeds
                    # Subtract transaction cost
                    cost = proceeds * transaction_cost
                    cash -= cost
                    units = 0
                    total_trades += 1

        # Increment time in market
        if units > 0:
            time_in_market_hours += 1

        # Calculate current equity
        equity = cash + units * row['Adj Close']
        equity_curve.append(equity)

    # Add equity curve to DataFrame
    df['Equity'] = equity_curve

    # Performance metrics
    final_equity = equity_curve[-1]
    total_return = (final_equity - initial_cash) / initial_cash * 100
    total_hours = len(df)
    trading_days = total_hours / 24  # Approximation
    if trading_days > 0:
        cagr = (final_equity / initial_cash) ** (1 / (trading_days / 252)) - 1
    else:
        cagr = 0
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    time_in_market = time_in_market_hours / total_hours * 100
    equity_series = pd.Series(equity_curve)
    hourly_returns = equity_series.pct_change().dropna()
    if hourly_returns.std() != 0:
        sharpe_ratio = np.sqrt(252 * 24) * hourly_returns.mean() / hourly_returns.std()
    else:
        sharpe_ratio = 0
    initial_price = df['Adj Close'].iloc[0]
    df['Buy_and_Hold'] = initial_cash * (df['Adj Close'] / initial_price)

    metrics = {
        'Final Equity': final_equity,
        'Total Return (%)': total_return,
        'CAGR (%)': cagr * 100,
        'Max Drawdown (%)': max_drawdown,
        'Sharpe Ratio': sharpe_ratio,
        'Total Trades': total_trades,
        'Time in Market (%)': time_in_market
    }

    return metrics, df


def plot_results(df, metrics, pair_name):
    """
    Plot Price with Signals, RSI, and Equity Curve.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing OHLC, RSI, Signals, and Equity.
    metrics : dict
        Performance metrics.
    pair_name : str
        Name of the currency pair.
    """
    fig, axs = plt.subplots(3, 1, figsize=(14, 18), sharex=True)

    # 1. Price chart with buy and sell signals
    buy_signals_plot = df[df['Signal'] == 'Buy']
    sell_signals_plot = df[df['Signal'] == 'Sell']

    axs[0].plot(df['Date'], df['Adj Close'], label='Price', color='blue')
    axs[0].scatter(buy_signals_plot['Date'], buy_signals_plot['Adj Close'],
                   marker='^', color='green', label='Buy Signal', alpha=1)
    axs[0].scatter(sell_signals_plot['Date'], sell_signals_plot['Adj Close'],
                   marker='v', color='red', label='Sell Signal', alpha=1)
    axs[0].set_ylabel('Price (Adj Close)')
    axs[0].set_title(f'Price Chart with Buy and Sell Signals for {pair_name}')
    axs[0].legend()
    axs[0].grid(True)

    # 2. RSI chart
    axs[1].plot(df['Date'], df['RSI'], label='RSI(2)', color='purple')
    axs[1].axhline(20, color='green', linestyle='--', alpha=0.7)
    axs[1].axhline(90, color='red', linestyle='--', alpha=0.7)
    axs[1].set_ylabel('RSI(2)')
    axs[1].set_title('Relative Strength Index (RSI)')
    axs[1].legend()
    axs[1].grid(True)

    # 3. Equity curve comparison: Strategy vs Buy-and-Hold
    axs[2].plot(df['Date'], df['Equity'], label='RSI Strategy Equity', color='blue')
    axs[2].plot(df['Date'], df['Buy_and_Hold'], label='Buy-and-Hold Equity', color='orange')
    axs[2].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    axs[2].set_ylabel('Equity (in dollars)')
    axs[2].set_title(f'Equity Curve Comparison: RSI Strategy vs Buy-and-Hold for {pair_name}')
    axs[2].legend()
    axs[2].grid(True)

    # Format x-axis for datetime
    axs[2].xaxis.set_major_locator(mdates.AutoDateLocator())
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plot_filename = os.path.join(plots_dir, f"{pair_name.replace('/', '_')}_backtest_performance.pdf")
    plt.savefig(plot_filename, format='pdf')
    plt.close()

    print(f"Plot saved to {plot_filename}")


def process_pair(file):
    """
    Process a single currency pair: load data, backtest, plot, and return metrics.

    Parameters:
    -----------
    file : str
        Filename of the CSV file.

    Returns:
    --------
    dict
        Performance metrics including Currency Pair.
    """
    pair_name = file.replace('_hourly.csv', '').replace('_', '/')
    file_path = os.path.join(data_dir, file)
    print(f"\nProcessing {pair_name}...")

    try:
        # Load and prepare data
        df = load_and_prepare_data(file_path)

        # Check if RSI has enough data points
        if df['RSI'].isnull().all():
            print(f"RSI calculation failed for {pair_name}. Skipping...")
            return None

        # Run backtest
        metrics, df = backtest_strategy(df)

        # Add Currency Pair to metrics
        metrics['Currency Pair'] = pair_name

        # Plot results
        plot_results(df, metrics, pair_name)

        return metrics

    except Exception as e:
        print(f"An error occurred while processing {pair_name}: {e}")
        return None

def main_backtest():
    """
    Main function to run backtest on all currency pairs.
    """
    # List all CSV files in the data directory
    data_files = [f for f in os.listdir(data_dir) if f.endswith('_hourly.csv')]

    if not data_files:
        print(f"No CSV files found in {data_dir}. Please ensure your data is correctly placed.")
        return

    # Initialize a list to store performance metrics for all pairs
    all_metrics = []

    # Use multiprocessing for faster processing
    pool = Pool(processes=os.cpu_count())
    results = pool.map(process_pair, data_files)
    pool.close()
    pool.join()

    # Collect all metrics
    all_metrics = [res for res in results if res is not None]

    if not all_metrics:
        print("No valid backtest results to compile.")
        return

    # Create a DataFrame from all_metrics
    performance_df = pd.DataFrame(all_metrics)

    # Reorder columns for better readability and sort by 'Final Equity' descending
    performance_df = performance_df[['Currency Pair', 'Final Equity', 'Total Return (%)', 
                                     'CAGR (%)', 'Max Drawdown (%)', 'Sharpe Ratio', 
                                     'Total Trades', 'Time in Market (%)']]

    # Sort the DataFrame by 'Final Equity' in descending order
    performance_df = performance_df.sort_values(by='Final Equity', ascending=False).reset_index(drop=True)

    # Save performance metrics to CSV
    metrics_file = os.path.join(results_dir, 'performance_metrics_sorted_by_final_equity.csv')
    performance_df.to_csv(metrics_file, index=False)

    print(f"\nAll performance metrics saved to {metrics_file}")

    # Display the performance metrics
    print("\nConsolidated Performance Metrics (Sorted by Final Equity):")
    print(performance_df)

if __name__ == "__main__":
    main_backtest()
