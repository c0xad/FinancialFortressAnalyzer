import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

def train_advanced_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1)  # Set verbose=-1 to suppress warnings
    }

    param_grids = {
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        },
        'LightGBM': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'num_leaves': [31, 63],
            'min_child_samples': [20, 50]
        }
    }

    best_models = {}

    for name, model in models.items():
        print(f"Training {name}...")
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        best_models[name] = grid_search.best_estimator_

        y_pred = best_models[name].predict(X_test_scaled)
        print(f"{name} Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}")
        print(f"F1 Score: {f1_score(y_test, y_pred, zero_division=0):.4f}\n")

    return best_models, scaler

def predict_fortress_balance_sheet_advanced(models, scaler, features):
    scaled_features = scaler.transform(features)
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict_proba(scaled_features)[0][1]
    return predictions