import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def get_historical_data(ticker, period='5y'):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    return hist

def safe_divide(a, b):
    return a / b.replace(0, np.nan)

def calculate_historical_ratios(ticker, period='5y'):
    stock = yf.Ticker(ticker)
    balance_sheet = stock.balance_sheet
    income_statement = stock.financials
    cash_flow = stock.cashflow

    print("Balance Sheet columns:", balance_sheet.columns)
    print("Balance Sheet index:", balance_sheet.index)
    print("Income Statement columns:", income_statement.columns)
    print("Income Statement index:", income_statement.index)
    print("Cash Flow columns:", cash_flow.columns)
    print("Cash Flow index:", cash_flow.index)

    ratios = pd.DataFrame()

    # Current Ratio
    if 'Total Current Assets' in balance_sheet.index and 'Total Current Liabilities' in balance_sheet.index:
        ratios['Current Ratio'] = safe_divide(balance_sheet.loc['Total Current Assets'], balance_sheet.loc['Total Current Liabilities'])
    else:
        print("Warning: Unable to calculate Current Ratio")

    # Quick Ratio
    if 'Total Current Assets' in balance_sheet.index and 'Inventory' in balance_sheet.index and 'Total Current Liabilities' in balance_sheet.index:
        ratios['Quick Ratio'] = safe_divide(balance_sheet.loc['Total Current Assets'] - balance_sheet.loc['Inventory'], balance_sheet.loc['Total Current Liabilities'])
    else:
        print("Warning: Unable to calculate Quick Ratio")

    # Debt to Equity
    if 'Total Debt' in balance_sheet.index and 'Total Stockholder Equity' in balance_sheet.index:
        ratios['Debt to Equity'] = safe_divide(balance_sheet.loc['Total Debt'], balance_sheet.loc['Total Stockholder Equity'])
    else:
        print("Warning: Unable to calculate Debt to Equity")

    # ROA
    if 'Net Income' in income_statement.index and 'Total Assets' in balance_sheet.index:
        ratios['ROA'] = safe_divide(income_statement.loc['Net Income'], balance_sheet.loc['Total Assets'])
    else:
        print("Warning: Unable to calculate ROA")

    # ROE
    if 'Net Income' in income_statement.index and 'Total Stockholder Equity' in balance_sheet.index:
        ratios['ROE'] = safe_divide(income_statement.loc['Net Income'], balance_sheet.loc['Total Stockholder Equity'])
    else:
        print("Warning: Unable to calculate ROE")

    return ratios

def perform_comparative_analysis(main_ticker, peer_tickers):
    main_ratios = calculate_historical_ratios(main_ticker)
    peer_ratios = {ticker: calculate_historical_ratios(ticker) for ticker in peer_tickers}

    comparison = pd.DataFrame()
    for ratio in main_ratios.columns:
        comparison.loc[main_ticker, ratio] = main_ratios[ratio].iloc[-1] if not main_ratios[ratio].empty else np.nan
        for peer in peer_tickers:
            comparison.loc[peer, ratio] = peer_ratios[peer][ratio].iloc[-1] if ratio in peer_ratios[peer].columns and not peer_ratios[peer][ratio].empty else np.nan

    return comparison

def calculate_risk_metrics(ticker, period='5y'):
    hist = get_historical_data(ticker, period)
    returns = hist['Close'].pct_change().dropna()

    risk_metrics = {
        'Volatility': returns.std() * np.sqrt(252),  # Annualized volatility
        'VaR_95': np.percentile(returns, 5),  # 95% Value at Risk
        'Max_Drawdown': (hist['Close'] / hist['Close'].cummax() - 1).min()
    }

    return risk_metrics

def generate_advanced_report(ticker, peer_tickers):
    print(f"\nAdvanced Analysis for {ticker}")
    
    # Historical Trend Analysis
    historical_ratios = calculate_historical_ratios(ticker)
    print("\nHistorical Financial Ratios:")
    print(historical_ratios)

    # Anomaly Detection
    if not historical_ratios.empty:
        anomalies = detect_anomalies(historical_ratios)
        print("\nAnomalies Detected:")
        print(historical_ratios[anomalies])
    else:
        print("\nUnable to perform anomaly detection due to insufficient data.")

    # Comparative Analysis
    comparison = perform_comparative_analysis(ticker, peer_tickers)
    print("\nComparative Analysis:")
    print(comparison)

    # Risk Assessment
    risk_metrics = calculate_risk_metrics(ticker)
    print("\nRisk Metrics:")
    for metric, value in risk_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Percentile Ranking
    for column in comparison.columns:
        values = comparison[column].dropna()
        if not values.empty and ticker in values.index:
            percentile = percentileofscore(values, values.loc[ticker])
            print(f"{ticker} is in the {percentile:.2f}th percentile for {column}")
        else:
            print(f"Unable to calculate percentile for {column}")

def detect_anomalies(data, contamination=0.1):
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data)

    # Normalize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_imputed)

    clf = IsolationForest(contamination=contamination, random_state=42)
    anomalies = clf.fit_predict(data_scaled)
    return anomalies == -1  # True for anomalies, False for normal data