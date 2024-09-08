import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Normalize the data
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data_imputed), columns=data_imputed.columns)

    return data_scaled, scaler

def engineer_features(data):
    # Add some financial ratios as features
    if 'Total Debt' in data.columns and 'Total Equity' in data.columns:
        data['Debt_to_Equity'] = data['Total Debt'] / data['Total Equity']
    if 'EBIT' in data.columns and 'Interest Expense' in data.columns:
        data['Interest_Coverage'] = data['EBIT'] / data['Interest Expense']
    if 'Net Income' in data.columns and 'Revenue' in data.columns:
        data['Profit_Margin'] = data['Net Income'] / data['Revenue']
    if 'Revenue' in data.columns and 'Total Assets' in data.columns:
        data['Asset_Turnover'] = data['Revenue'] / data['Total Assets']
    if 'Net Income' in data.columns and 'Total Equity' in data.columns:
        data['ROE'] = data['Net Income'] / data['Total Equity']
    
    return data