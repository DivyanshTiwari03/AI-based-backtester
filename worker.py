import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from llama_index.legacy.llms.ollama import Ollama
from typing import Dict
import pandas_ta as ta

class StrategyTester:
    def __init__(self, initial_capital: float = 10000):
        self.data = None
        self.strategy_code = ""
        self.initial_capital = initial_capital
        self.symbol = ""
        self.logs = []

    def load_data(self, symbol: str, start_date: str, end_date: str):
        self.symbol = symbol
        self.data = yf.download(symbol, start=start_date, end=end_date)
        if self.data.empty:
            raise ValueError(f"No data available for symbol {symbol}")
        # Keep the datetime index
        self.data.index.name = 'Date'

    def generate_strategy_code(self, strategy_prompt: str) -> str:
        llm = Ollama(model="codestral:latest", request_timeout=120.0)
        prompt = f"""
        Generate Python code for a trading strategy based on the following description:
        {strategy_prompt}
        
        The code should define a function called 'execute_strategy' that takes a pandas DataFrame 
        with a DatetimeIndex named 'Date', and columns 'Open', 'High', 'Low', 'Close', and 'Volume', 
        and returns a pandas Series of trade signals (-1 for sell, 0 for hold, 1 for buy) with the same index as the input DataFrame. 
        Output only the code and no explanation. Assume all imports are already there. You can use pandas_ta library for indicators, it's been imported as "ta".

        Important guidelines:
        1. Use proper datetime indexing. The index is already a DatetimeIndex named 'Date'.
        2. Initialize the signals Series with zeros for all dates.
        3. Use 'signals' (plural) for the Series name, not 'signal'.
        4. Ensure the function returns 'signals' at the end.
        5. Use proper Python indentation.
        6. Output only the code and no explanation.
        7. Use variable for in_position, if in position only look for sell, do not have consecutive buys or sells.
        8. Make sure the strategy is general and not specific to any particular stock.
        9. Include print statements to show the logic being applied (e.g., "Buy signal generated on date df.index[i]").

        Here's a template to start with:

        def execute_strategy(df):
            signals = pd.Series(0, index=df.index, name='Signal')
            in_position = False
            for i in range(len(df)):
                # Your strategy logic here
                # Use df.iloc[i] to access the current row
                # Modify signals.iloc[i] to -1, 0, or 1 based on your strategy
            return signals
        """
        response = str(llm.complete(prompt))
        start_marker = "def"
        end_marker = "return signals"
        code_block = response[response.find(start_marker):response.find(end_marker)+len(end_marker)].strip()
        self.strategy_code = code_block
        return self.strategy_code

    def backtest_strategy(self) -> Dict:
        try:
            safe_locals = {
                'pd': pd,
                'np': np,
                'DataFrame': pd.DataFrame,
                'Series': pd.Series,
                'ta': ta
            }
            
            exec(self.strategy_code, globals(), safe_locals)
            signals = safe_locals['execute_strategy'](self.data)
            
            if len(signals) != len(self.data):
                raise ValueError("Strategy returned incorrect number of signals")
            
            self.data['Signal'] = signals
            self.data['Capital'] = self.initial_capital
            self.data['Shares'] = 0.0
            self.data['Portfolio_Value'] = self.initial_capital

            in_position = False
            for i in range(1, len(self.data)):
                prev_capital = self.data['Capital'].iloc[i-1]
                prev_shares = self.data['Shares'].iloc[i-1]
                current_price = self.data['Close'].iloc[i]
                
                if self.data['Signal'].iloc[i] == 1 and not in_position:  # Buy signal
                    shares_to_buy = prev_capital / current_price
                    cost = shares_to_buy * current_price
                    self.data.loc[self.data.index[i], 'Shares'] = shares_to_buy
                    self.data.loc[self.data.index[i], 'Capital'] = prev_capital - cost
                    in_position = True
                    self.logs.append(f"Buy executed on {self.data.index[i]}: Bought {shares_to_buy:.8f} shares at {current_price:.2f}")
                elif self.data['Signal'].iloc[i] == -1 and in_position:  # Sell signal
                    sale_proceeds = prev_shares * current_price
                    self.data.loc[self.data.index[i], 'Shares'] = 0
                    self.data.loc[self.data.index[i], 'Capital'] = prev_capital + sale_proceeds
                    in_position = False
                    self.logs.append(f"Sell executed on {self.data.index[i]}: Sold {prev_shares:.8f} shares at {current_price:.2f}")
                else:  # Hold
                    self.data.loc[self.data.index[i], 'Shares'] = prev_shares
                    self.data.loc[self.data.index[i], 'Capital'] = prev_capital
                
                self.data.loc[self.data.index[i], 'Portfolio_Value'] = self.data['Capital'].iloc[i] + (self.data['Shares'].iloc[i] * current_price)

            # Calculate buy-and-hold strategy
            initial_shares = self.initial_capital / self.data['Close'].iloc[0]
            self.data['Buy_Hold_Value'] = initial_shares * self.data['Close']

            # Calculate performance metrics
            total_return = (self.data['Portfolio_Value'].iloc[-1] - self.initial_capital) / self.initial_capital
            buy_hold_return = (self.data['Buy_Hold_Value'].iloc[-1] - self.initial_capital) / self.initial_capital
            strategy_returns = self.data['Portfolio_Value'].pct_change().dropna()
            sharpe_ratio = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
            
            return {
                'Symbol': self.symbol,
                'Total Strategy Return': total_return,
                'Buy and Hold Return': buy_hold_return,
                'Sharpe Ratio': sharpe_ratio,
                'Final Strategy Value': self.data['Portfolio_Value'].iloc[-1],
                'Final Buy and Hold Value': self.data['Buy_Hold_Value'].iloc[-1],
                'Number of Trades': sum(abs(self.data['Signal']))//2
            }
        except Exception as e:
            return {'Error': str(e)}

    def visualize_strategy(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
        
        # Plot price and signals
        ax1.plot(self.data.index, self.data['Close'], label='Close Price', alpha=0.5)
        ax1.scatter(self.data.index[self.data['Signal'] == 1], 
                    self.data.loc[self.data['Signal'] == 1, 'Close'], 
                    marker='^', color='g', label='Buy Signal')
        ax1.scatter(self.data.index[self.data['Signal'] == -1], 
                    self.data.loc[self.data['Signal'] == -1, 'Close'], 
                    marker='v', color='r', label='Sell Signal')
        ax1.set_title(f'Price and Trading Signals for {self.symbol}')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.legend()

        # Plot Portfolio Value
        ax2.plot(self.data.index, self.data['Portfolio_Value'], label='Strategy Portfolio Value')
        ax2.plot(self.data.index, self.data['Buy_Hold_Value'], label='Buy and Hold Value')
        ax2.set_title(f'Portfolio Value for {self.symbol} (Initial Capital: ${self.initial_capital:,.0f})')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend()
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        plt.tight_layout()
        return fig

def main():
    st.title("Trading Strategy Tester")

    # User inputs
    symbol = st.text_input("Enter the stock symbol (e.g., AAPL, GOOGL):")
    start_date = st.date_input("Start date", value=pd.to_datetime("2022-01-01"))
    end_date = st.date_input("End date", value=pd.to_datetime("2023-12-31"))
    strategy_prompt = st.text_area("Describe your trading strategy in English:")
    
    if st.button("Run Backtest"):
        tester = StrategyTester()
        
        # Load data
        tester.load_data(symbol, start_date, end_date)
        
        # Generate strategy code
        strategy_code = tester.generate_strategy_code(strategy_prompt)
        st.subheader("Generated Strategy Code:")
        st.code(strategy_code)
        
        # Backtest strategy
        results = tester.backtest_strategy()
        
        # Display results
        st.subheader("Backtesting Results:")
        if 'Error' in results:
            st.error(f"An error occurred: {results['Error']}")
        else:
            for metric, value in results.items():
                if isinstance(value, float):
                    st.write(f"{metric}: {value:.4f}")
                else:
                    st.write(f"{metric}: {value}")
        
        # Visualize strategy
        st.subheader("Strategy Visualization")
        fig = tester.visualize_strategy()
        st.pyplot(fig)

        # Display logs in a collapsible section
        st.subheader("Execution Logs")
        with st.expander("Click to view detailed execution logs"):
            for log in tester.logs:
                st.write(log)

if __name__ == "__main__":
    main()