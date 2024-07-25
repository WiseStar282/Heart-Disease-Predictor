import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Asumsi bahwa df sudah memiliki kolom 'target'
    X = df[['age', 'gender', 'impluse', 'pressurehight', 'pressurelow', 'glucose', 'kcm', 'troponin']]
    y = df['class']
    return X, y

def split_data(X, y, test_size=0.3):
    return train_test_split(X, y, test_size=test_size, random_state=42)
