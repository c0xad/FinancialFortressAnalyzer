FinancialFortressAnalyzer is a comprehensive Python-based tool for analyzing company financials, predicting balance sheet strength, and providing advanced financial insights. It combines traditional financial ratio analysis with machine learning techniques to offer a holistic view of a company's financial health.

Features:
Financial Data Retrieval: Automatically fetches financial statements and stock data for specified companies.  Calculates key financial ratios including Current Ratio, Quick Ratio, Debt-to-Equity Ratio, Return on Assets (ROA), and Return on Equity (ROE).
Ratio Analysis: Calculates key financial ratios including liquidity, solvency, and profitability metrics.
Fortress Balance Sheet Prediction: Utilizes advanced machine learning models to predict the likelihood of a company having a "fortress" balance sheet.
Comparative Analysis: Compares a company's financial metrics with its peers.
Historical Trend Analysis: Analyzes historical financial ratios to identify trends.
Risk Assessment: Calculates key risk metrics including volatility, Value at Risk (VaR), and maximum drawdown.
Sentiment Analysis: Incorporates news sentiment to provide a more comprehensive view of a company's standing.
Advanced Visualizations: Generates detailed charts and graphs for easy interpretation of financial data and predictions.
Valuation Metrics: Computes important valuation metrics such as Price-to-Earnings (P/E) Ratio and Price-to-Book (P/B) Ratio.

How to Use
Clone the repository
Install required dependencies: pip install -r requirements.txt
Set up your API keys for financial data and news sentiment analysis
Run the main script: python main.py

The script will analyze Apple Inc. (AAPL) and compare it with Microsoft (MSFT), Alphabet (GOOGL), and Amazon (AMZN) by default. You can modify the main_ticker and peer_tickers variables in main.py to analyze different companies.

Key Functions
analyze_company(ticker, peer_tickers, news_api_key): Main function to perform comprehensive analysis on a company.

calculate_ratios(balance_sheet, income_statement, cash_flow): Calculates financial ratios from financial statements.

is_fortress_balance_sheet(ratios): Determines if a company has a fortress balance sheet based on predefined criteria.

train_model(historical_data): Trains machine learning models on historical financial data.

predict_fortress_balance_sheet(ratios, valuation_metrics, models, scaler): Predicts the likelihood of a fortress balance sheet using trained models.

create_visualizations(ticker, ratios, valuation_metrics, avg_prediction, historical_ratios, comparison, risk_metrics): Generates comprehensive visualizations of the analysis.Key Functions

analyze_company(ticker, peer_tickers, news_api_key): Main function to perform comprehensive analysis on a company.

calculate_ratios(balance_sheet, income_statement, cash_flow): Calculates financial ratios from financial statements.

is_fortress_balance_sheet(ratios): Determines if a company has a fortress balance sheet based on predefined criteria.

train_model(historical_data): Trains machine learning models on historical financial data.

predict_fortress_balance_sheet(ratios, valuation_metrics, models, scaler): Predicts the likelihood of a fortress balance sheet using trained models.

create_visualizations(ticker, ratios, valuation_metrics, avg_prediction, historical_ratios, comparison, risk_metrics): Generates comprehensive visualizations of the analysis.


Disclaimer
This tool is for educational and research purposes only. Always conduct your own due diligence before making investment decisions.
