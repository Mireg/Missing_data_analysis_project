from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import lightgbm as lgb
import numpy as np

def tune_xgboost(X_train, y_train, cv=5):
    param_grid = {
        'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05, 0.1],
        'max_depth': [3, 5, 7, 8, 9, 10],
        'n_estimators': [100, 200, 300, 400, 500, 550, 600, 700],
        'min_child_weight': [1, 2, 3, 4, 5, 7],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    }
    
    model = xgb.XGBRegressor(random_state=42)
    search = RandomizedSearchCV(
        model, param_grid, n_iter=20, 
        scoring='neg_root_mean_squared_error',
        cv=cv, random_state=42, n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

def tune_lightgbm(X_train, y_train, cv=5):
    param_grid = {
        'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05, 0.1],
        'num_leaves': [31, 42, 50, 63, 70, 80],
        'n_estimators': [100, 200, 300, 400, 500, 550, 600, 700],
        'min_child_samples': [5, 10, 20, 30, 50],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    }
    
    model = lgb.LGBMRegressor(random_state=42)
    search = RandomizedSearchCV(
        model, param_grid, n_iter=20,
        scoring='neg_root_mean_squared_error',
        cv=cv, random_state=42, n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_