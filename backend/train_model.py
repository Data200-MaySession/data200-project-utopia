import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Define paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'moviesDb.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'logistic_regression_model.joblib')
COLUMNS_PATH = os.path.join(MODELS_DIR, 'model_columns.joblib')

def train_and_save_model():
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please ensure 'moviesDb.csv' is in the 'backend/data/' directory.")
        return

    print("Preparing data for training...")
    predictors = ['budget', 'runtime', 'year', 'vote_average', 'vote_count', 'certification_US', 'genre', 'country']
    
    # Ensure all predictor columns exist
    missing_predictors = [p for p in predictors if p not in df.columns]
    if missing_predictors:
        print(f"Error: The following predictor columns are missing from the dataset: {missing_predictors}")
        return

    # Handle potential NaN values in categorical predictors before get_dummies
    # For simplicity, we'll fill with a placeholder string like 'Unknown' or mode if appropriate
    # This ensures get_dummies doesn't create NaN columns or error out
    for col in ['certification_US', 'genre', 'country']:
        if df[col].isnull().any():
            print(f"Warning: NaN values found in categorical column '{col}'. Filling with 'Unknown'.")
            df[col] = df[col].fillna('Unknown')

    x_dummy = pd.get_dummies(df[predictors], columns=['certification_US', 'genre', 'country'])
    y = df['success'].values

    print(f"Training Logistic Regression model with C=1000, solver='newton-cg'...")
    # Using the best parameters identified in your classification.py for Logistic Regression
    # C: 1000, solver: newton-cg. Max_iter might be needed for convergence.
    log_reg_model = LogisticRegression(C=1000, solver='newton-cg', max_iter=1000) # Increased max_iter
    log_reg_model.fit(x_dummy, y)

    print(f"Saving trained model to {MODEL_PATH}...")
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    joblib.dump(log_reg_model, MODEL_PATH)

    print(f"Saving model columns to {COLUMNS_PATH}...")
    model_columns = list(x_dummy.columns)
    joblib.dump(model_columns, COLUMNS_PATH)

    print("Model training and saving complete.")

if __name__ == '__main__':
    train_and_save_model()
