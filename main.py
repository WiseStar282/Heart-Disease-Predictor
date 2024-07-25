import os
from src.preprocessing import load_data, preprocess_data, split_data
from src.training import train_model, save_model
from src.evaluation import evaluate_model
from src.predict import predict, predict_probabilities
import numpy as np

def main():
    # Path data
    data_path = 'data/Heart Attack.csv'
    
    # Load and preprocess data
    df = load_data(data_path)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train model
    model = train_model(X_train, y_train)
    save_model(model, 'random_forest_model.pkl')
    
    # Evaluate model
    conf_matrix, class_report = evaluate_model(model, X_test, y_test)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

if __name__ == "__main__":
    main()
