import yfinance as yf

def get_financial_data(ticker):
    company = yf.Ticker(ticker)
    balance_sheet = company.balance_sheet.T  # Transpose the dataframe
    income_statement = company.financials.T  # Transpose the dataframe
    cash_flow = company.cashflow.T  # Transpose the dataframe
    return balance_sheet, income_statement, cash_flow

def get_stock_data(ticker):
    company = yf.Ticker(ticker)
    return company.info