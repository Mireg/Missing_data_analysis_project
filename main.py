import pandas as pd
import numpy as np
from src.imputers import ImputationMethods
from src.models import get_models, get_optimized_models
from src.pipeline import ModelPipeline
from src.utils import prepare_data, load_data, save_results
import xgboost as xgb
from lightgbm import LGBMRegressor

def main():
    # Load data
    train_df, test_df = load_data("data/pzn-rent-train-processed.csv", "data/pzn-rent-test-processed.csv")
    
    # Initial data preparation
    #drop_columns = ['ad_title', 'date_activ', 'date_modif', 'date_expire']
    #train_df.drop(columns=drop_columns, inplace=True)
    #test_df.drop(columns=drop_columns, inplace=True)
    
    # Get all unique quarters from both datasets
    all_quarters = pd.concat([train_df['quarter'], test_df['quarter']]).unique()
    
    # Create dummy variables for both datasets with consistent columns
    train_quarters = pd.get_dummies(train_df['quarter'], prefix='quarter')
    test_quarters = pd.get_dummies(test_df['quarter'], prefix='quarter')
    
    # Add missing columns to each dataset
    missing_in_train = set(test_quarters.columns) - set(train_quarters.columns)
    missing_in_test = set(train_quarters.columns) - set(test_quarters.columns)
    
    for col in missing_in_train:
        train_quarters[col] = 0
    for col in missing_in_test:
        test_quarters[col] = 0
        
    # Ensure columns are in the same order
    train_quarters = train_quarters[sorted(train_quarters.columns)]
    test_quarters = test_quarters[sorted(test_quarters.columns)]
    
    # Drop original quarter column and add encoded columns
    train_df = train_df.drop(columns=['quarter'])
    test_df = test_df.drop(columns=['quarter'])
    
    # Add encoded quarters back (dropping one category to avoid multicollinearity)
    quarter_cols = sorted(train_quarters.columns)[1:]  # Skip first column
    train_df = pd.concat([train_df, train_quarters[quarter_cols]], axis=1)
    test_df = pd.concat([test_df, test_quarters[quarter_cols]], axis=1)
    
    # Prepare data for modeling
    X_train, y_train, numeric_features, categorical_features = prepare_data(train_df, 'price')
    X_test, _, _, _ = prepare_data(test_df)
    
    # Initialize imputation methods
    imputer = ImputationMethods()
    imputation_methods = {
        'no_imputation': imputer.get_imputers()['no_imputation']
        #'simple_most_frequent': imputer.get_imputers()['simple_most_frequent'],
        #'knn': imputer.get_imputers()['knn'],
        #'simple_mean': imputer.get_imputers()['simple_mean'],
        #'simple_median': imputer.get_imputers()['simple_median'],
        #'advanced_iterative': imputer.get_imputers()['advanced_iterative'],
        #'ensemble': imputer.get_imputers()['ensemble'],
        #'lightgbm': imputer.get_imputers()['lightgbm'],
        #'hybrid': imputer.get_imputers()['hybrid']
    }
    
    # Get base and optimized models
    base_models = get_models()
    optimized_models = get_optimized_models()
    
    # Combine models
    selected_models = {
        'xgboost': base_models['xgboost'],
        'lightgbm': base_models['lightgbm'],
        'xgboost_optimized': optimized_models['xgboost_optimized'],
        'lightgbm_optimized': optimized_models['lightgbm_optimized']
    }
    
    # Create and run pipeline
    print("\nStarting model pipeline...")
    pipeline = ModelPipeline(imputation_methods, selected_models)
    results = pipeline.run_pipeline(X_train, y_train, X_test, numeric_features, categorical_features)
    
    # Save results
    print("\nSaving results...")
    save_results(results)
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()