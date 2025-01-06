import pandas as pd
import os
import time
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np

def prepare_data(df, target_column=None):
    """Prepare data and identify column types"""
    if target_column and target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df
        y = None
        
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    return X, y, numeric_features, categorical_features

def load_data(train_path, test_path):
    """Load and check datasets"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Print dataset information
    for name, df in [("Training", train_df), ("Test", test_df)]:
        missing_stats = df.isnull().sum()
        missing_percentages = (missing_stats / len(df)) * 100
        
        print(f"\n{name} Dataset Missing Value Statistics:")
        print("-" * (len(name) + 28))
        for column in df.columns:
            if missing_stats[column] > 0:
                print(f"{column}: {missing_stats[column]} missing values ({missing_percentages[column]:.2f}%)")
    
    return train_df, test_df

def save_results(results, output_dir='predictions'):
    """Save predictions and comparison results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual predictions and create comparison summary
    summary_data = []
    for approach_name, result in results.items():
        # Save predictions
        predictions_df = pd.DataFrame({
            'price': result['predictions']
        })
        predictions_df.to_csv(f'{output_dir}/{approach_name}_predictions.csv', index=False)
        
        # Add to summary
        summary_data.append({
            'approach': approach_name,
            'cv_rmse_mean': result['cv_rmse_mean'],
            'cv_rmse_std': result['cv_rmse_std'],
            'training_time': result['training_time']
        })
    
    # Save and display summary
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('cv_rmse_mean')
    summary_df.to_csv(f'{output_dir}/approach_comparison.csv', index=False)
    
    print("\nApproach Comparison:")
    print(summary_df.to_string(index=False))