def is_fortress_balance_sheet(ratios):
    if ratios is None:
        return False, {}
    
    criteria = {
        'Current Ratio': ratios.get('Current Ratio', 0) > 1.5,
        'Quick Ratio': ratios.get('Quick Ratio', 0) > 1,
        'Cash Ratio': ratios.get('Cash Ratio', 0) > 0.5,
        'Debt-to-Equity Ratio': ratios.get('Debt-to-Equity Ratio', float('inf')) < 0.5,
        'Debt-to-Assets Ratio': ratios.get('Debt-to-Assets Ratio', float('inf')) < 0.3,
        'Equity Ratio': ratios.get('Equity Ratio', 0) > 0.5,
        'Operating Cash Flow Ratio': ratios.get('Operating Cash Flow Ratio', 0) > 0,
        'Interest Coverage Ratio': ratios.get('Interest Coverage Ratio', 0) > 3
    }
    return all(criteria.values()), criteria