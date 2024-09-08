import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from advanced_ml_models import train_advanced_models, predict_fortress_balance_sheet_advanced
from data_preprocessor import preprocess_data, engineer_features

def engineer_features(data):

    engineered_data = data.copy()
    
    if 'Total Debt' in data.columns and 'Total Equity' in data.columns:
        engineered_data['Debt_to_Equity'] = data['Total Debt'] / data['Total Equity']
    
    if 'EBIT' in data.columns and 'Interest Expense' in data.columns:
        engineered_data['Interest_Coverage'] = data['EBIT'] / data['Interest Expense']
    
    if 'Revenue' in data.columns and 'Total Assets' in data.columns:
        engineered_data['Asset_Turnover'] = data['Revenue'] / data['Total Assets']
    
    if 'Net Income' in data.columns and 'Revenue' in data.columns:
        engineered_data['Profit_Margin'] = data['Net Income'] / data['Revenue']
    
    return engineered_data

def prepare_data(ratios, valuation_metrics):
    """
    Prepare data for the model, including feature engineering.
    """
    features = {**ratios, **valuation_metrics}
    df = pd.DataFrame([features])
    df = engineer_features(df)
    return df

def load_historical_data():
    """
    Load historical financial data for multiple companies.
    In a real-world scenario, this would fetch data from a database or API.
    """
    # Simulating historical data for demonstration purposes
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    data = pd.DataFrame(np.random.randn(n_samples, n_features), 
                        columns=[f'feature_{i}' for i in range(n_features)])
    
    # Add some correlated features
    data['Total Debt'] = np.random.randint(1000, 10000, n_samples)
    data['Total Equity'] = np.random.randint(5000, 20000, n_samples)
    data['EBIT'] = np.random.randint(500, 5000, n_samples)
    data['Interest Expense'] = np.random.randint(100, 1000, n_samples)
    data['Revenue'] = np.random.randint(10000, 100000, n_samples)
    data['Total Assets'] = np.random.randint(20000, 200000, n_samples)
    data['Net Income'] = np.random.randint(1000, 10000, n_samples)

    # Create target variable (fortress balance sheet or not)
    data['target'] = (data['Total Debt'] / data['Total Equity'] < 0.5) & \
                     (data['EBIT'] / data['Interest Expense'] > 3) & \
                     (data['Net Income'] / data['Revenue'] > 0.1)
    
    return data

def train_model(historical_data):
    # Ensure there's a 'target' column
    if 'target' not in historical_data.columns:
        # You'll need to define how to create the target variable
        # This is just an example; adjust according to your needs
        historical_data['target'] = (historical_data['Return on Assets (ROA)'] > historical_data['Return on Assets (ROA)'].mean()).astype(int)

    X = historical_data.drop('target', axis=1)
    y = historical_data['target']

    X_engineered = engineer_features(X)
    X_preprocessed, _ = preprocess_data(X_engineered)  # Unpack the tuple

    # Ensure X and y have the same number of samples
    X_preprocessed = X_preprocessed[:len(y)]
    y = y[:len(X_preprocessed)]

    models, scaler = train_advanced_models(X_preprocessed, y)
    return models, scaler

def predict_fortress_balance_sheet(ratios, valuation_metrics, models, scaler):
    features = {**ratios, **valuation_metrics}
    df = pd.DataFrame([features])
    df_engineered = engineer_features(df)
    
    # Ensure the columns match those used during training
    expected_columns = scaler.feature_names_in_
    missing_columns = set(expected_columns) - set(df_engineered.columns)
    extra_columns = set(df_engineered.columns) - set(expected_columns)
    
    # Add missing columns with zero values
    for col in missing_columns:
        df_engineered[col] = 0
    
    # Remove extra columns
    df_engineered = df_engineered.drop(columns=extra_columns)
    
    # Reorder columns to match the order used during training
    df_engineered = df_engineered[expected_columns]
    
    return predict_fortress_balance_sheet_advanced(models, scaler, df_engineered)
