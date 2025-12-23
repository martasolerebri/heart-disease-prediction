import pandas as pd
import joblib
import os
from src import (
    median_mode_imputation_minus_nine,
    median_mode_imputation_question_mark,
    run_complete_pipeline
)

DATA_RAW_TRAIN = './data/raw/train.csv'
DATA_RAW_TEST = './data/raw/test.csv'
MODEL_OUTPUT_PATH = './models/heart_disease_model.pkl'
LABEL_VARS = ['sex', 'fbs', 'exang', 'slope']
ONEHOT_VARS = ['cp', 'restecg', 'ca', 'thal']

def main():
    print("--- Starting Heart Disease Prediction Pipeline ---")
    
    if not os.path.exists(DATA_RAW_TRAIN):
        print(f"Error: {DATA_RAW_TRAIN} not found.")
        return

    train = pd.read_csv(DATA_RAW_TRAIN)
    test = pd.read_csv(DATA_RAW_TEST)

    best_params = [{
        'solver': ['lbfgs'],
        'penalty': ['l2'],
        'C': [0.1],
        'class_weight': [None],
        'max_iter': [10000]
    }]

    print("Training model...")
    (
        X_train_scaled, 
        y_train, 
        final_model, 
        feature_names, 
        scaler,
        grid_search, 
        cv_score
    ) = run_complete_pipeline(
        train, 
        test,
        impute_minus_nine_func=median_mode_imputation_minus_nine,
        impute_question_func=median_mode_imputation_question_mark,
        label_vars=LABEL_VARS,
        onehot_vars=ONEHOT_VARS,
        param_grid=best_params,
        use_feature_engineering=True,
        submission=True
    )

    print(f"Saving model to {MODEL_OUTPUT_PATH}...")
    joblib.dump(final_model, MODEL_OUTPUT_PATH)
    joblib.dump(scaler, 'models/scaler.pkl') 
    joblib.dump(feature_names.tolist(), 'models/features_list.pkl')
    
    print(f"Pipeline finished. CV Accuracy: {cv_score:.4f}")

if __name__ == "__main__":
    main()