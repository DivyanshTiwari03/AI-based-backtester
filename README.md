# AI Trading Strategy Backtester

A Streamlit application that converts plain English trading strategies into executable Python code using the Codestral LLM model and performs backtesting with detailed visualization and analysis.

## Description

This application allows users to:
- Input trading strategies in plain English
- Automatically convert strategies to Python code using the Codestral LLM
- Backtest strategies against historical stock data
- Visualize trading signals and portfolio performance
- Compare results against buy-and-hold strategy
- View detailed execution logs and performance metrics

## Prerequisites

- Python 3.8+
- Ollama installed locally with the Codestral model
- Required Python packages

## Installation

1. First, install Ollama following instructions at [Ollama Installation](https://ollama.ai/download)

2. Pull the Codestral model:
```bash
ollama pull codestral:latest
```

3. Clone this repository:
```bash
git clone https://github.com/yourusername/ai-trading-backtester.git
cd ai-trading-backtester
```

4. Install required Python packages:
```bash
pip install -r requirements.txt
```

## Requirements

Create a `requirements.txt` file with the following dependencies:
```
streamlit
yfinance
pandas
numpy
matplotlib
llama-index
pandas-ta
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run worker.py
```

2. In the web interface:
   - Enter a stock symbol (e.g., AAPL, GOOGL)
   - Select date range for backtesting
   - Describe your trading strategy in plain English
   - Click "Run Backtest" to execute

Example strategy description:
```
Buy when the 20-day moving average crosses above the 50-day moving average, and sell when it crosses below. Only take one position at a time.
```

## Features

- Real-time code generation from English descriptions
- Historical data fetching using yfinance
- Comprehensive backtesting engine
- Performance metrics calculation:
  - Total return
  - Buy-and-hold comparison
  - Sharpe ratio
  - Number of trades
- Interactive visualizations:
  - Price chart with buy/sell signals
  - Portfolio value comparison
- Detailed execution logs

## Notes

- The application uses the Codestral LLM model locally through Ollama
- Ensure Ollama is running before starting the application
- The backtesting assumes zero transaction costs and perfect execution
- Historical data is fetched from Yahoo Finance

## Limitations

- Limited to stock data available on Yahoo Finance
- Backtesting results are theoretical and don't account for real-world factors
- Strategy generation depends on the Codestral model's capabilities
- No support for complex order types or position sizing
