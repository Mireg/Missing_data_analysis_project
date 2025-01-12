from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor

def get_models(random_state=42):
    return {
        'random_forest': RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=random_state,
            n_jobs=-1
        ),
        'xgboost': xgb.XGBRegressor(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1
        ),
        'lightgbm': LGBMRegressor(
            n_estimators=500,
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            random_state=random_state,
            n_jobs=-1
        )
    }

def get_optimized_models(random_state=42):
    """Get optimized model configurations based on quick evaluation results"""
    return {
        'xgboost_optimized': xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=3,
            random_state=random_state,
            n_jobs=-1
        ),
        'lightgbm_optimized': LGBMRegressor(
            n_estimators=500,
            num_leaves=63,
            learning_rate=0.05,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            min_child_samples=10,
            random_state=random_state,
            n_jobs=-1
        )
    }