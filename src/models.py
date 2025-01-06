from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor

def get_models(random_state=42):
    return {
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
        'xgboost': xgb.XGBRegressor(random_state=random_state, n_jobs=-1),
        'lightgbm': LGBMRegressor(random_state=random_state, n_jobs=-1)
    }