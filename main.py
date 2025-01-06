from src.imputers import ImputationMethods
from src.models import get_models
from src.pipeline import ModelPipeline
from src.utils import prepare_data, load_data, save_results

def main():
    try:
        # Load data
        train_df, test_df = load_data("train_dataset.csv", "test_dataset.csv")
        
        # Prepare data
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
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()