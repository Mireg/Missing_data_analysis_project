import pandas as pd
from src.imputers import ImputationMethods
from src.models import get_models
from src.pipeline import ModelPipeline
from src.utils import prepare_data, load_data, save_results

def main():
        # Load data
        train_df, test_df = load_data("data/pzn-rent-train.csv", "data/pzn-rent-test.csv")
        train_df, test_df = load_data("data/pzn-rent-train-cleaned_v1.csv", "data/pzn-rent-test-cleaned_v1.csv")

        # Prepare data
        train_df.drop(columns = ['ad_title', 'date_activ', 'date_modif', 'date_expire'], inplace=True)
        test_df.drop(columns = ['ad_title', 'date_activ', 'date_modif', 'date_expire'], inplace=True)
        train_df = pd.get_dummies(train_df, columns=['quarter'], drop_first=True)
        test_df = pd.get_dummies(test_df, columns=['quarter'], drop_first=True)
        
        X_train, y_train, numeric_features, categorical_features = prepare_data(train_df, 'price')
        X_test, _, _, _ = prepare_data(test_df)
        
        # Initialize components
        imputation_methods = ImputationMethods().get_imputers()
        prediction_models = get_models()
        
        # Create and run pipeline
        pipeline = ModelPipeline(imputation_methods, prediction_models)
        results = pipeline.run_pipeline(X_train, y_train, X_test, numeric_features, categorical_features)
        
        # Save results
        save_results(results)

if __name__ == "__main__":
    main()