def calculate_valuation_metrics(stock_data, income_statement, balance_sheet):
    try:
        market_price = stock_data.get('currentPrice', None)
        eps = stock_data.get('trailingEps', None)
        book_value_per_share = stock_data.get('bookValue', None)

        valuation_metrics = {}

        if market_price is not None and eps is not None and eps != 0:
            valuation_metrics['Price-to-Earnings (P/E) Ratio'] = market_price / eps

        if market_price is not None and book_value_per_share is not None and book_value_per_share != 0:
            valuation_metrics['Price-to-Book (P/B) Ratio'] = market_price / book_value_per_share

        return valuation_metrics
    except Exception as e:
        print(f"Error calculating valuation metrics: {e}")
        return {}