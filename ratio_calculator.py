import numpy as np

def calculate_ratios(balance_sheet, income_statement, cash_flow):
    ratios = {}
    try:
        print("Balance Sheet:")
        print(balance_sheet)
        print("\nIncome Statement:")
        print(income_statement)
        print("\nCash Flow:")
        print(cash_flow)

        # Use the most recent data (first row)
        bs = balance_sheet.iloc[0]
        is_ = income_statement.iloc[0]
        cf = cash_flow.iloc[0]

        # Current Ratio
        if 'Total Current Assets' in bs.index and 'Total Current Liabilities' in bs.index:
            ratios['Current Ratio'] = bs['Total Current Assets'] / bs['Total Current Liabilities']
        else:
            print("Warning: Unable to calculate Current Ratio due to missing data")

        # Quick Ratio
        if 'Total Current Assets' in bs.index and 'Inventory' in bs.index and 'Total Current Liabilities' in bs.index:
            ratios['Quick Ratio'] = (bs['Total Current Assets'] - bs['Inventory']) / bs['Total Current Liabilities']
        else:
            print("Warning: Unable to calculate Quick Ratio due to missing data")

        # Debt-to-Equity Ratio
        if 'Total Debt' in bs.index and 'Total Stockholder Equity' in bs.index:
            ratios['Debt-to-Equity Ratio'] = bs['Total Debt'] / bs['Total Stockholder Equity']
        else:
            print("Warning: Unable to calculate Debt-to-Equity Ratio due to missing data")

        # Return on Assets (ROA)
        if 'Net Income' in is_.index and 'Total Assets' in bs.index:
            ratios['Return on Assets (ROA)'] = is_['Net Income'] / bs['Total Assets']
        else:
            print("Warning: Unable to calculate Return on Assets due to missing data")

        # Return on Equity (ROE)
        if 'Net Income' in is_.index and 'Total Stockholder Equity' in bs.index:
            ratios['Return on Equity (ROE)'] = is_['Net Income'] / bs['Total Stockholder Equity']
        else:
            print("Warning: Unable to calculate Return on Equity due to missing data")

        print("\nCalculated Ratios:")
        for ratio, value in ratios.items():
            print(f"{ratio}: {value}")

        return ratios

    except Exception as e:
        print(f"Error calculating ratios: {e}")
        return {}

def calculate_additional_ratios(balance_sheet, income_statement, cash_flow):
    try:
        net_income = income_statement.loc['Net Income', income_statement.columns[0]]
        total_assets = balance_sheet.loc['Total Assets', balance_sheet.columns[0]]
        total_equity = balance_sheet.loc['Total Stockholder Equity', balance_sheet.columns[0]] if 'Total Stockholder Equity' in balance_sheet.index else balance_sheet.loc['Stockholders Equity', balance_sheet.columns[0]]

        additional_ratios = {
            'Return on Assets (ROA)': net_income / total_assets,
            'Return on Equity (ROE)': net_income / total_equity,
        }
        return additional_ratios
    except Exception as e:
        print(f"Error calculating additional ratios: {e}")
        return {}