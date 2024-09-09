from advanced_analysis import calculate_historical_ratios, perform_comparative_analysis, calculate_risk_metrics, generate_advanced_report
from data_fetcher import get_financial_data, get_stock_data
from ratio_calculator import calculate_ratios
from balance_sheet_analyzer import is_fortress_balance_sheet
from valuation_metrics import calculate_valuation_metrics
from predictive_model import train_model, predict_fortress_balance_sheet, load_historical_data
from visualizer import create_visualizations
from sentiment_analyzer import get_company_sentiment, interpret_sentiment
from advanced_visualizations import create_interactive_visualizations, plot_feature_importance


def analyze_company(ticker, peer_tickers, news_api_key):
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

    # Predict using advanced models
    historical_data = load_historical_data()
    models, scaler, feature_importance = train_model(historical_data)
    predictions = predict_fortress_balance_sheet(ratios, valuation_metrics, models, scaler)

    print("\nAdvanced Model Predictions:")
    for model_name, prediction in predictions.items():
        print(f"{model_name}: {prediction:.4f}")

    # Perform sentiment analysis
    sentiment_score = get_company_sentiment(ticker, news_api_key)
    sentiment = interpret_sentiment(sentiment_score)
    print(f"\nCurrent Sentiment: {sentiment} (Score: {sentiment_score:.2f})")

    # Add advanced analysis
    historical_ratios = calculate_historical_ratios(ticker)
    comparison = perform_comparative_analysis(ticker, peer_tickers)
    risk_metrics = calculate_risk_metrics(ticker)
    generate_advanced_report(ticker, peer_tickers)

    # Create visualizations
    if ratios and valuation_metrics:
        avg_prediction = sum(predictions.values()) / len(predictions)
        create_visualizations(ticker, ratios, valuation_metrics, avg_prediction, historical_ratios, comparison, risk_metrics)
        create_interactive_visualizations(ticker, ratios, valuation_metrics, avg_prediction, historical_ratios, comparison, risk_metrics)
    else:
        print("Unable to create visualizations due to missing data.")

    # Plot feature importance
    plot_feature_importance(feature_importance, historical_data.drop('target', axis=1).columns)

# Example usage
main_ticker = "AAPL"  # Apple Inc.
peer_tickers = ["MSFT", "GOOGL", "AMZN"]  # Microsoft, Alphabet, Amazon
news_api_key = "54303364278e474fb5dd36ec81eb692454303364278e474fb5dd36ec81eb6924"  # Replace with your actual News API key
analyze_company(main_ticker, peer_tickers, news_api_key)