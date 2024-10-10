import os
import glob
import logging
from itertools import product
from multiprocessing import Pool
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import shutil

# Directories and configurations
data_dir = 'currency_data'
results_dir = 'backtest_results'
batch_results_dir = os.path.join(results_dir, 'batch_results')
optimized_plots_dir = os.path.join(results_dir, 'optimized_plots')
os.makedirs(batch_results_dir, exist_ok=True)
os.makedirs(optimized_plots_dir, exist_ok=True)
TRAIN_END_DATE = '2024-09-10 00:00:00'  

# Set up logging
logging.basicConfig(
    filename=os.path.join(results_dir, 'optimization.log'),
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Define optimization parameter grid
param_grid = {
    'rsi_length': [2, 5, 10],
    'oversold': [10, 20, 30],
    'overbought': [70, 80, 90],
    'risk_percent': [0.01, 0.02, 0.03],
    'stop_loss_percent': [0.01, 0.02, 0.03],
    'take_profit_percent': [0.05, 0.1, 0.15]
}

def load_and_prepare_data(file_path, rsi_length, oversold, overbought):
    """Load data, preprocess, and calculate RSI within a specified date range."""
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter data to be within the desired date range
    start_date = '2024-08-04 23:00:00'
    end_date = '2024-09-27 21:00:00'
    df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
    
    # Proceed with the rest of the processing
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.rename(
        columns={
            'timestamp': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close'
        },
        inplace=True
    )
    df['Adj Close'] = df['Close']
    df['RSI'] = ta.rsi(df['Adj Close'], length=rsi_length)

    # Generate buy/sell signals based on RSI
    df['Signal'] = None
    df.loc[df['RSI'] < oversold, 'Signal'] = 'Buy'
    df.loc[df['RSI'] > overbought, 'Signal'] = 'Sell'

    return df


def split_data(df):
    """Split data into training and testing sets."""
    train_data = df[df['Date'] < TRAIN_END_DATE]
    test_data = df[df['Date'] >= TRAIN_END_DATE]
    print(f"Training data size: {len(train_data)}")
    print(f"Testing data size: {len(test_data)}")
    return train_data, test_data


# def backtest_strategy(
#     df,
#     risk_percent,
#     stop_loss_percent,
#     take_profit_percent,
#     initial_cash=100_000
# ):
#     """Backtest RSI strategy and return performance metrics."""
#     cash = initial_cash
#     units = 0
#     equity = initial_cash
#     equity_curve = []
#     total_trades = 0
#     time_in_market_hours = 0
#     started = False
#     purchase_price = 0

#     buy_signals = df[df['Signal'] == 'Buy'].index
#     sell_signals = df[df['Signal'] == 'Sell'].index

#     for i, row in df.iterrows():
#         try:
#             if not started and i in buy_signals:
#                 position_risk = equity * risk_percent
#                 stop_loss = row['Adj Close'] * (1 - stop_loss_percent)
#                 risk_per_unit = row['Adj Close'] - stop_loss
#                 units_to_buy = int(position_risk // risk_per_unit) if risk_per_unit > 0 else 0

#                 if units_to_buy > 0 and cash >= units_to_buy * row['Adj Close']:
#                     units += units_to_buy
#                     cash -= units_to_buy * row['Adj Close']
#                     cash -= units_to_buy * row['Adj Close'] * 0.0002
#                     total_trades += 1
#                     purchase_price = row['Adj Close']
#                     started = True

#             elif started:
#                 if i in buy_signals and units == 0:
#                     position_risk = equity * risk_percent
#                     stop_loss = row['Adj Close'] * (1 - stop_loss_percent)
#                     risk_per_unit = row['Adj Close'] - stop_loss
#                     units_to_buy = int(position_risk // risk_per_unit) if risk_per_unit > 0 else 0

#                     if units_to_buy > 0 and cash >= units_to_buy * row['Adj Close']:
#                         units += units_to_buy
#                         cash -= units_to_buy * row['Adj Close']
#                         cash -= units_to_buy * row['Adj Close'] * 0.0002
#                         total_trades += 1
#                         purchase_price = row['Adj Close']

#                 elif i in sell_signals and units > 0:
#                     proceeds = units * row['Adj Close']
#                     cash += proceeds - proceeds * 0.0002
#                     units = 0
#                     total_trades += 1

#                 elif units > 0:
#                     if row['Adj Close'] >= purchase_price * (1 + take_profit_percent):
#                         proceeds = units * row['Adj Close']
#                         cash += proceeds - proceeds * 0.0002
#                         units = 0
#                         total_trades += 1

#                     elif row['Adj Close'] <= purchase_price * (1 - stop_loss_percent):
#                         proceeds = units * row['Adj Close']
#                         cash += proceeds - proceeds * 0.0002
#                         units = 0
#                         total_trades += 1

#             if units > 0:
#                 time_in_market_hours += 1

#             equity = cash + units * row['Adj Close']
#             equity_curve.append(equity)
#         except Exception as e:
#             logging.error(f"Error during backtesting at index {i}: {e}", exc_info=True)
#             return None, df

#     final_equity = equity_curve[-1]
#     total_return = (final_equity - initial_cash) / initial_cash * 100
#     equity_series = pd.Series(equity_curve)
#     returns = equity_series.pct_change()
#     if returns.std() != 0:
#         sharpe_ratio = (
#             np.sqrt(252 * 24) * returns.mean() / returns.std()
#         )
#     else:
#         sharpe_ratio = 0

#     metrics = {
#         'Final Equity': final_equity,
#         'Total Return (%)': total_return,
#         'Sharpe Ratio': sharpe_ratio,
#         'Total Trades': total_trades,
#         'Time in Market (%)': (time_in_market_hours / len(df)) * 100
#     }

#     df = df.copy()
#     df['Equity'] = equity_curve  # Add equity curve to DataFrame

#     return metrics, df

def backtest_strategy(
    df,
    risk_percent,
    stop_loss_percent,
    take_profit_percent,
    initial_cash=100_000
):
    """Backtest RSI strategy and return performance metrics."""
    cash = initial_cash
    units = 0
    equity = initial_cash
    equity_curve = []
    total_trades = 0
    profitable_trades = 0
    drawdowns = []
    time_in_market_hours = 0
    started = False
    purchase_price = 0
    peak_equity = initial_cash

    buy_signals = df[df['Signal'] == 'Buy'].index
    sell_signals = df[df['Signal'] == 'Sell'].index

    for i, row in df.iterrows():
        try:
            if not started and i in buy_signals:
                position_risk = equity * risk_percent
                stop_loss = row['Adj Close'] * (1 - stop_loss_percent)
                risk_per_unit = row['Adj Close'] - stop_loss
                units_to_buy = int(position_risk // risk_per_unit) if risk_per_unit > 0 else 0

                if units_to_buy > 0 and cash >= units_to_buy * row['Adj Close']:
                    units += units_to_buy
                    cash -= units_to_buy * row['Adj Close']
                    cash -= units_to_buy * row['Adj Close'] * 0.0002
                    total_trades += 1
                    purchase_price = row['Adj Close']
                    started = True

            elif started:
                if i in buy_signals and units == 0:
                    position_risk = equity * risk_percent
                    stop_loss = row['Adj Close'] * (1 - stop_loss_percent)
                    risk_per_unit = row['Adj Close'] - stop_loss
                    units_to_buy = int(position_risk // risk_per_unit) if risk_per_unit > 0 else 0

                    if units_to_buy > 0 and cash >= units_to_buy * row['Adj Close']:
                        units += units_to_buy
                        cash -= units_to_buy * row['Adj Close']
                        cash -= units_to_buy * row['Adj Close'] * 0.0002
                        total_trades += 1
                        purchase_price = row['Adj Close']

                elif i in sell_signals and units > 0:
                    proceeds = units * row['Adj Close']
                    cash += proceeds - proceeds * 0.0002
                    units = 0
                    total_trades += 1
                    if row['Adj Close'] > purchase_price:
                        profitable_trades += 1

                elif units > 0:
                    if row['Adj Close'] >= purchase_price * (1 + take_profit_percent):
                        proceeds = units * row['Adj Close']
                        cash += proceeds - proceeds * 0.0002
                        units = 0
                        total_trades += 1
                        profitable_trades += 1

                    elif row['Adj Close'] <= purchase_price * (1 - stop_loss_percent):
                        proceeds = units * row['Adj Close']
                        cash += proceeds - proceeds * 0.0002
                        units = 0
                        total_trades += 1

            if units > 0:
                time_in_market_hours += 1

            # Update equity and peak equity
            equity = cash + units * row['Adj Close']
            peak_equity = max(peak_equity, equity)
            equity_curve.append(equity)
        except Exception as e:
            logging.error(f"Error during backtesting at index {i}: {e}", exc_info=True)
            return None, df

    # Calculating drawdowns
    equity_series = pd.Series(equity_curve)
    drawdown_series = (equity_series / equity_series.cummax()) - 1
    max_drawdown = drawdown_series.min() * 100  # Convert to percentage
    average_drawdown = drawdown_series[drawdown_series < 0].mean() * 100  # Only negative values

    # Final equity and metrics calculations
    final_equity = equity_curve[-1]
    total_return = (final_equity - initial_cash) / initial_cash * 100
    returns = equity_series.pct_change().dropna()
    volatility = returns.std() * np.sqrt(252 * 24) * 100  # Annualized volatility
    downside_deviation = returns[returns < 0].std() * np.sqrt(252 * 24) * 100

    # Metrics with fallback to avoid division by zero
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 24) if returns.std() != 0 else 0
    sortino_ratio = (returns.mean() / downside_deviation) * np.sqrt(252 * 24) if downside_deviation != 0 else 0
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0

    metrics = {
        'Final Equity': final_equity,
        'Total Return (%)': total_return,
        'Equity Peak': peak_equity,
        'Volatility (%)': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown (%)': max_drawdown,
        'Average Drawdown (%)': average_drawdown,
        'Total Trades': total_trades,
        'Win Rate (%)': win_rate,
        'Time in Market (%)': (time_in_market_hours / len(df)) * 100
    }

    df = df.copy()
    df['Equity'] = equity_curve  # Add equity curve to DataFrame

    return metrics, df



def optimize_and_test(file_path):
    """Optimize parameters on training set, then test on testing set."""
    pair_name = os.path.basename(file_path)
    pair_name = pair_name.replace('_hourly.csv', '').replace('_', '/')
    logging.info(f"Starting optimization for {pair_name}")

    # Check if file exists
    if not os.path.exists(file_path):
        logging.warning(f"File {file_path} not found, skipping {pair_name}.")
        return None

    try:
        best_metrics = None
        best_params = None

        # Iterate over all combinations of parameters
        for params in product(*param_grid.values()):
            param_dict = dict(zip(param_grid.keys(), params))
            logging.info(f"Evaluating parameters: {param_dict}")

            # Load and prepare data with current parameters
            df = load_and_prepare_data(
                file_path,
                rsi_length=param_dict['rsi_length'],
                oversold=param_dict['oversold'],
                overbought=param_dict['overbought']
            )
            train_data, test_data = split_data(df)

            train_metrics, _ = backtest_strategy(
                train_data.copy(),
                risk_percent=param_dict['risk_percent'],
                stop_loss_percent=param_dict['stop_loss_percent'],
                take_profit_percent=param_dict['take_profit_percent']
            )

            if train_metrics is None:
                continue  # Skip to next parameter set if error occurred

            if best_metrics is None or \
               train_metrics['Final Equity'] > best_metrics['Final Equity']:
                best_metrics = train_metrics
                best_params = param_dict
                logging.info(
                    f"New best on training set: {best_params} -> "
                    f"Final Equity: {train_metrics['Final Equity']}"
                )

        if best_params is None:
            logging.warning(f"No valid parameters found for {pair_name}.")
            return None

        logging.info(
            f"Best parameters from training for {pair_name}: {best_params}"
        )

        # Load and prepare data with best parameters
        df = load_and_prepare_data(
            file_path,
            rsi_length=best_params['rsi_length'],
            oversold=best_params['oversold'],
            overbought=best_params['overbought']
        )
        train_data, test_data = split_data(df)

        # Evaluate on the testing set
        test_metrics, df_with_equity = backtest_strategy(
            test_data,
            risk_percent=best_params['risk_percent'],
            stop_loss_percent=best_params['stop_loss_percent'],
            take_profit_percent=best_params['take_profit_percent']
        )

        if test_metrics is None:
            logging.warning(f"Testing failed for {pair_name} with best parameters.")
            return None

        logging.info(
            f"Testing results for {pair_name}: "
            f"Final Equity: {test_metrics['Final Equity']}"
        )

        return {
            'Currency Pair': pair_name,
            'Best Metrics': best_metrics,
            'Best Parameters': best_params,
            'Testing Metrics': test_metrics,
            'DataFrame': df_with_equity
        }

    except Exception as e:
        logging.error(f"Error processing {pair_name}: {e}", exc_info=True)
        return None


def main():
    # Get list of currency data files
    currency_files = glob.glob(os.path.join(data_dir, '*.csv'))

    # Use multiprocessing Pool to process files in parallel
    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(optimize_and_test, currency_files)

    # Filter out None results (in case some files failed)
    results = [r for r in results if r is not None]

    # If no results, exit
    if not results:
        logging.error("No valid results obtained from optimization.")
        return

    # Sort results by 'Testing Metrics'['Final Equity'] in descending order
    sorted_results = sorted(results, key=lambda x: x['Testing Metrics']['Final Equity'], reverse=True)

    # Create a DataFrame to store all performance metrics for each currency pair
    metrics_data = []
    for res in sorted_results:
        metrics_data.append({
            'Currency Pair': res['Currency Pair'],
            'Final Equity': res['Testing Metrics']['Final Equity'],
            'Total Return (%)': res['Testing Metrics']['Total Return (%)'],
            'Equity Peak': res['Testing Metrics']['Equity Peak'],
            'Volatility (%)': res['Testing Metrics']['Volatility (%)'],
            'Sharpe Ratio': res['Testing Metrics']['Sharpe Ratio'],
            'Sortino Ratio': res['Testing Metrics']['Sortino Ratio'],
            'Max Drawdown (%)': res['Testing Metrics']['Max Drawdown (%)'],
            'Average Drawdown (%)': res['Testing Metrics']['Average Drawdown (%)'],
            'Total Trades': res['Testing Metrics']['Total Trades'],
            'Win Rate (%)': res['Testing Metrics']['Win Rate (%)'],
            'Time in Market (%)': res['Testing Metrics']['Time in Market (%)'],
            'Best Parameters': res['Best Parameters']
        })

    # Convert list of dictionaries to a DataFrame and sort it
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.sort_values(by='Final Equity', ascending=False, inplace=True)

    # Save the DataFrame to a CSV file in the results directory
    metrics_csv_path = os.path.join(results_dir, 'performance_metrics.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)

    # Logging the CSV save action
    logging.info(f"Performance metrics saved to {metrics_csv_path}")

    # Define a batch folder for top results based on the timestamp
    batch_folder_name = os.path.join(batch_results_dir, 'batch_' + pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(batch_folder_name, exist_ok=True)

    # Get top 3 results and store in batch folder
    for idx, res in enumerate(sorted_results[:3], start=1):
        pair_name = res['Currency Pair']
        metrics = res['Testing Metrics']
        df = res['DataFrame']
        best_params = res['Best Parameters']

        logging.info(f"Top {idx} - {pair_name}")
        logging.info(f"Best Parameters: {best_params}")
        logging.info(f"Testing Metrics: {metrics}")

        # Create subfolder for each top result within the batch folder
        top_result_folder = os.path.join(batch_folder_name, f'top_{idx}_{pair_name.replace("/", "_")}')
        os.makedirs(top_result_folder, exist_ok=True)

        # Define PDF file path for the current top result
        pdf_filename = os.path.join(top_result_folder, 'result_plots.pdf')
        optimized_pdf_filename = os.path.join(optimized_plots_dir, f'top_{idx}_{pair_name.replace("/", "_")}_plots.pdf')

        # Create a PDF file to save all plots
        with PdfPages(pdf_filename) as pdf:
            fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

            # Identify the start and end dates
            start_date = df['Date'].iloc[0]
            end_date = df['Date'].iloc[-1]
            split_date = pd.to_datetime(TRAIN_END_DATE)

            # Plot price with buy/sell signals
            axes[0].plot(df['Date'], df['Adj Close'], label='Price')
            buy_signals = df[df['Signal'] == 'Buy']
            sell_signals = df[df['Signal'] == 'Sell']
            axes[0].scatter(buy_signals['Date'], buy_signals['Adj Close'], marker='^', color='green', label='Buy Signal', s=100)
            axes[0].scatter(sell_signals['Date'], sell_signals['Adj Close'], marker='v', color='red', label='Sell Signal', s=100)
            axes[0].set_title(f'{pair_name} Price')
            axes[0].set_ylabel('Price')
            axes[0].legend()

            # Plot RSI with overbought and oversold levels and split line
            axes[1].plot(df['Date'], df['RSI'], label='RSI', color='orange')
            axes[1].axhline(y=best_params['oversold'], color='green', linestyle='--', label='Oversold')
            axes[1].axhline(y=best_params['overbought'], color='red', linestyle='--', label='Overbought')
            axes[1].set_title('RSI Indicator')
            axes[1].set_ylabel('RSI')
            axes[1].legend()

            # Plot Equity Curve with split line
            axes[2].plot(df['Date'], df['Equity'], label='Equity', color='purple')
            axes[2].set_title('Equity Curve')
            axes[2].set_ylabel('Equity')
            axes[2].legend()

            # Set custom x-axis ticks to only show start, split, and end dates
            for ax in axes:
                ax.set_xticks([start_date, split_date, end_date])
                ax.set_xticklabels([
                    start_date.strftime('%Y-%m-%d %H:%M'), 
                    split_date.strftime('%Y-%m-%d %H:%M'), 
                    end_date.strftime('%Y-%m-%d %H:%M')
                ])
                plt.setp(ax.get_xticklabels(), rotation=45)

            plt.tight_layout()

            # Save the figure to the PDF
            pdf.savefig(fig)
            plt.close(fig)

        # Copy the PDF to the optimized_plots directory
        shutil.copy(pdf_filename, optimized_pdf_filename)

        # Save metrics to a text file within the result folder
        metrics_filename = os.path.join(top_result_folder, 'metrics.txt')
        with open(metrics_filename, 'w') as f:
            f.write(f"Currency Pair: {pair_name}\n")
            f.write(f"Best Parameters: {best_params}\n")
            f.write("Testing Metrics:\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")

    logging.info("Optimization and testing complete.")



if __name__ == '__main__':
    main()
