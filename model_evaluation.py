#!/usr/bin/env python3

# model_evaluation.py - Evaluate the machine learning model for the Movie Success Predictor
# Metrics: Accuracy, Precision, Recall, F1 Score, and Confusion Matrix

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'moviesDb.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'backend', 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'logistic_regression_model.joblib')
COLUMNS_PATH = os.path.join(MODELS_DIR, 'model_columns.joblib')

def evaluate_model():
    print("Loading data...")
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATASET_PATH}")
        return
    
    print("Loading model and model columns...")
    try:
        model = joblib.load(MODEL_PATH)
        model_columns = joblib.load(COLUMNS_PATH)
    except FileNotFoundError:
        print(f"Error: Model files not found at {MODEL_PATH} or {COLUMNS_PATH}")
        return
    
    # Prepare data for evaluation
    print("Preparing data for evaluation...")
    predictors = ['budget', 'runtime', 'year', 'vote_average', 'vote_count', 'certification_US', 'genre', 'country']
    
    # Process categorical features
    for col in ['certification_US', 'genre', 'country']:
        if df[col].isnull().any():
            df[col] = df[col].fillna('Unknown')
    
    # Create dummy variables
    X = pd.get_dummies(df[predictors], columns=['certification_US', 'genre', 'country'])
    y = df['success'].values
    
    # Align test data with model's expected columns
    X = X.reindex(columns=model_columns, fill_value=0)
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    print("\n--- Model Evaluation Results ---")
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Precision
    precision = precision_score(y_test, y_pred)
    print(f"Precision: {precision:.4f}")
    
    # Recall
    recall = recall_score(y_test, y_pred)
    print(f"Recall: {recall:.4f}")
    
    # F1 Score
    f1 = f1_score(y_test, y_pred)
    print(f"F1 Score: {f1:.4f}")
    
    # Classification Report
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=['Flop', 'Hit'])
    print(report)
    
    # Confusion Matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Flop', 'Predicted Hit'],
                yticklabels=['Actual Flop', 'Actual Hit'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix: True Hits vs False Flops')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")
    
    # Additional analysis: feature importance
    if hasattr(model, 'coef_'):
        print("\nTop 10 Important Features:")
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': np.abs(model.coef_[0])
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False).head(10)
        print(feature_importance)

if __name__ == '__main__':
    evaluate_model()
    print("\nModel evaluation complete!") 