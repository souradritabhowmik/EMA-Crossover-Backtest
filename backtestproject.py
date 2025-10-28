# ema_crossover_backtest.py

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

class EMACrossoverBacktest:
    def __init__(self, ticker, start_date, end_date, short_window=10, long_window=50, initial_capital=10000):
        """
        Initialize the EMA Crossover Backtest.
        :param ticker: Stock ticker (e.g., "AAPL")
        :param start_date: Start date (YYYY-MM-DD)
        :param end_date: End date (YYYY-MM-DD)
        :param short_window: Short-term EMA period
        :param long_window: Long-term EMA period
        :param initial_capital: Starting capital for the backtest
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.short_window = short_window
        self.long_window = long_window
        self.initial_capital = initial_capital
        self.data = None
        self.signals = None
        self.results = None

    def fetch_data(self):
        """Download historical stock data using yfinance."""
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        self.data['Short_EMA'] = self.data['Close'].ewm(span=self.short_window, adjust=False).mean()
        self.data['Long_EMA'] = self.data['Close'].ewm(span=self.long_window, adjust=False).mean()
        self.data.dropna(inplace=True)

    def generate_signals(self):
        """Generate buy/sell signals based on EMA crossover."""
        self.signals = pd.DataFrame(index=self.data.index)
        self.signals['signal'] = 0
        self.signals['signal'][self.short_window:] = np.where(
            self.data['Short_EMA'][self.short_window:] > self.data['Long_EMA'][self.short_window:], 1, 0
        )
        self.signals['positions'] = self.signals['signal'].diff()

    def backtest_strategy(self):
        """Run the backtest and compute portfolio value."""
        self.results = self.data.copy()
        self.results['positions'] = self.signals['positions']
        self.results['cash'] = self.initial_capital
        self.results['holdings'] = 0
        self.results['total'] = self.initial_capital

        position = 0
        cash = self.initial_capital

        for i in range(len(self.results)):
            if self.results['positions'].iloc[i] == 1:  # Buy signal
                position = cash / self.results['Close'].iloc[i]
                cash = 0
            elif self.results['positions'].iloc[i] == -1:  # Sell signal
                cash = position * self.results['Close'].iloc[i]
                position = 0
            holdings = position * self.results['Close'].iloc[i]
            self.results['holdings'].iloc[i] = holdings
            self.results['cash'].iloc[i] = cash
            self.results['total'].iloc[i] = cash + holdings

    def plot_results(self):
        """Plot the stock price, EMAs, and portfolio value."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Price + EMA plot
        ax1.plot(self.data['Close'], label='Close Price', color='black')
        ax1.plot(self.data['Short_EMA'], label=f'Short EMA ({self.short_window})', color='blue')
        ax1.plot(self.data['Long_EMA'], label=f'Long EMA ({self.long_window})', color='red')
        ax1.plot(self.data.index[self.signals.positions == 1], 
                 self.data['Close'][self.signals.positions == 1], 
                 '^', markersize=10, color='green', label='Buy Signal')
        ax1.plot(self.data.index[self.signals.positions == -1], 
                 self.data['Close'][self.signals.positions == -1], 
                 'v', markersize=10, color='red', label='Sell Signal')
        ax1.set_title(f'{self.ticker} Price and EMA Crossover Signals')
        ax1.legend()
        ax1.grid()

        # Portfolio value plot
        ax2.plot(self.results['total'], label='Portfolio Value', color='purple')
        ax2.set_title('Portfolio Value Over Time')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend()
        ax2.grid()
        plt.show()

if __name__ == "__main__":
    # Example usage
    ticker = 'AAPL'
    start_date = '2022-01-01'
    end_date = '2023-01-01'

    backtest = EMACrossoverBacktest(ticker, start_date, end_date)
    backtest.fetch_data()
    backtest.generate_signals()
    backtest.backtest_strategy()
    backtest.plot_results()
    
    print(f"Final Portfolio Value: ${backtest.results['total'].iloc[-1]:.2f}")
