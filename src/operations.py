"""
Operations module for Job Placement Prediction MLOps Project
Contains utility functions for data processing, model training, and evaluation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score




def data_quality_check(df):
    """
    Perform data quality checks on the dataset
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        dict: Quality metrics including missing values and duplicates
    """
    quality_report = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    return quality_report


def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """
    Evaluate model performance with multiple metrics
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        y_pred_proba (array-like, optional): Predicted probabilities for ROC-AUC
        
    Returns:
        dict: Evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc_score'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics


def normalize_features(X_train, X_test, scaler):
    """
    Normalize features using provided scaler
    
    Args:
        X_train (array-like): Training features
        X_test (array-like): Testing features
        scaler: Scaler object (e.g., StandardScaler)
        
    Returns:
        tuple: (scaled_X_train, scaled_X_test)
    """
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def print_metrics(metrics):
    """
    Print evaluation metrics in a readable format
    
    Args:
        metrics (dict): Dictionary containing evaluation metrics
    """
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"\nConfusion Matrix:\n{np.array(metrics['confusion_matrix'])}")
    if 'roc_auc_score' in metrics:
        print(f"\nROC-AUC Score: {metrics['roc_auc_score']:.4f}")


def add(a, b):
    """
    Add two numbers
    
    Args:
        a (int/float): First number
        b (int/float): Second number
        
    Returns:
        int/float: Sum of a and b
    """
    return a + b


def multiply(a, b):
    """
    Multiply two numbers
    
    Args:
        a (int/float): First number
        b (int/float): Second number
        
    Returns:
        int/float: Product of a and b
    """
    return a * b


def subtract(a, b):
    """
    Subtract two numbers
    
    Args:
        a (int/float): First number
        b (int/float): Second number
        
    Returns:
        int/float: Difference of a and b
    """
    return a - b


def divide(a, b):
    """
    Divide two numbers
    
    Args:
        a (int/float): Numerator
        b (int/float): Denominator
        
    Returns:
        float: Quotient of a and b
        
    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
