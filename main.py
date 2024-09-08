from advanced_analysis import calculate_historical_ratios, perform_comparative_analysis, calculate_risk_metrics
from data_fetcher import get_financial_data, get_stock_data
from ratio_calculator import calculate_ratios
from balance_sheet_analyzer import is_fortress_balance_sheet
from valuation_metrics import calculate_valuation_metrics
from predictive_model import predict_fortress_balance_sheet
from visualizer import create_visualizations
from advanced_analysis import generate_advanced_report, calculate_historical_ratios, perform_comparative_analysis, calculate_risk_metrics

def analyze_company(ticker, peer_tickers):
    # Fetch financial data
    balance_sheet, income_statement, cash_flow = get_financial_data(ticker)
    stock_data = get_stock_data(ticker)

    # Calculate ratios and analyze balance sheet
    ratios = calculate_ratios(balance_sheet, income_statement, cash_flow)
    if not ratios:
        print(f"Unable to analyze balance sheet for {ticker} due to missing or inconsistent data.")
        return

    is_fortress, criteria = is_fortress_balance_sheet(ratios)

    # Calculate valuation metrics
    valuation_metrics = calculate_valuation_metrics(stock_data, income_statement, balance_sheet)

    # Predict future fortress balance sheet likelihood
    prediction = predict_fortress_balance_sheet(ratios, valuation_metrics)

    # Print results
    print(f"\nAnalysis for {ticker}:")
    print("Financial Ratios:")
    for ratio, value in ratios.items():
        print(f"{ratio}: {value:.2f}")
    
    print("\nFortress Balance Sheet Criteria:")
    for criterion, met in criteria.items():
        print(f"{criterion}: {'Met' if met else 'Not Met'}")
    
    print(f"\nConclusion: This {'is' if is_fortress else 'is not'} a fortress balance sheet.")

    print("\nValuation Metrics:")
    for metric, value in valuation_metrics.items():
        print(f"{metric}: {value:.2f}")

    print(f"\nPredicted likelihood of maintaining a fortress balance sheet: {prediction:.2f}")

    # Add advanced analysis
    historical_ratios = calculate_historical_ratios(ticker)
    comparison = perform_comparative_analysis(ticker, peer_tickers)
    risk_metrics = calculate_risk_metrics(ticker)
    generate_advanced_report(ticker, peer_tickers)

    # Create visualizations only if we have data
    if ratios and valuation_metrics:
        create_visualizations(ticker, ratios, valuation_metrics, prediction, historical_ratios, comparison, risk_metrics)
    else:
        print("Unable to create visualizations due to missing data.")

# Example usage
main_ticker = "AAPL"  # Apple Inc.
peer_tickers = ["MSFT", "GOOGL", "AMZN"]  # Microsoft, Alphabet, Amazon
analyze_company(main_ticker, peer_tickers)