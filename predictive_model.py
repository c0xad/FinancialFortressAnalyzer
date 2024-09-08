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

def engineer_features(data):
    """
    Create new features from existing ones, handling missing columns.
    """
    if 'Total Debt' in data.columns and 'Total Equity' in data.columns:
        data['Debt_to_Equity'] = data['Total Debt'] / data['Total Equity']
    
    if 'EBIT' in data.columns and 'Interest Expense' in data.columns:
        data['Interest_Coverage'] = data['EBIT'] / data['Interest Expense']
    
    if 'Revenue' in data.columns and 'Total Assets' in data.columns:
        data['Asset_Turnover'] = data['Revenue'] / data['Total Assets']
    
    if 'Net Income' in data.columns and 'Revenue' in data.columns:
        data['Profit_Margin'] = data['Net Income'] / data['Revenue']
    
    return data

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

def create_model():
    """
    Create a complex ensemble model with hyperparameter tuning.
    """
    # Define the preprocessing steps
    numeric_features = ['feature_' + str(i) for i in range(20)] + \
                       ['Total Debt', 'Total Equity', 'EBIT', 'Interest Expense', 
                        'Revenue', 'Total Assets', 'Net Income']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])

    # Create a pipeline with preprocessing, feature selection, and the final estimator
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(f_classif, k=10)),
        ('classifier', None)  # This will be set during grid search
    ])

    # Define the models and their specific hyperparameters
    param_grid = [
        {
            'classifier': [RandomForestClassifier(random_state=42)],
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [3, 5, 7]
        },
        {
            'classifier': [GradientBoostingClassifier(random_state=42)],
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [3, 5, 7]
        },
        {
            'classifier': [XGBClassifier(random_state=42)],
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [3, 5, 7]
        },
        {
            'classifier': [LogisticRegression(random_state=42)],
            'classifier__C': [0.1, 1, 10]
        },
        {
            'classifier': [SVC(random_state=42, probability=True)],
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['rbf', 'linear']
        }
    ]

    # Create the grid search object
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)

    return grid_search

def train_model(model, X, y):
    """
    Train the model using the provided data.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return model

def predict_fortress_balance_sheet(ratios, valuation_metrics):
    # Load historical data
    historical_data = load_historical_data()
    historical_data = engineer_features(historical_data)

    # Prepare the input data
    input_data = prepare_data(ratios, valuation_metrics)

    # Ensure input_data has the same columns as historical_data
    for col in historical_data.columns:
        if col not in input_data.columns:
            input_data[col] = np.nan

    # Create and train the model
    model = create_model()
    trained_model = train_model(model, historical_data.drop('target', axis=1), historical_data['target'])

    # Make prediction
    prediction = trained_model.predict_proba(input_data)[0][1]

    # Get feature importances
    feature_importances = None
    if hasattr(trained_model.best_estimator_.named_steps['classifier'], 'feature_importances_'):
        feature_importances = trained_model.best_estimator_.named_steps['classifier'].feature_importances_
    elif hasattr(trained_model.best_estimator_.named_steps['classifier'], 'coef_'):
        feature_importances = trained_model.best_estimator_.named_steps['classifier'].coef_[0]

    if feature_importances is not None:
        feature_names = trained_model.best_estimator_.named_steps['preprocessor'].get_feature_names_out()
        importances = sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True)
        print("\nTop 5 Important Features:")
        for name, importance in importances[:5]:
            print(f"{name}: {importance:.4f}")

    return prediction