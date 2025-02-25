import os
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import pickle

# Ensure logs directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

#logging config
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path):
    """
    Load data from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug(f"Data loaded and Nan's values are filled from {file_path}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing file {file_path}: {e}")
        raise
    except FileNotFoundError as e:
        logger.error(f"File {file_path} not found: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, param: dict) -> RandomForestClassifier:
    """
    Train a random forest classifier model.
    """
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of sample in X_train and y_train must be equal")
        logger.debug(f'Initializing RandomForest model with parameters: {param}')
        clf = RandomForestClassifier(n_estimators=param['n_estimators'], random_state=param['random_state'])
        logger.debug(f'Model training started with {X_train.shape[0]} samples')
        clf.fit(X_train, y_train)
        logger.debug('Model training completed')
        return clf
    
    except ValueError as e:
        logger.error(f"Error in model training: {e}")
        raise

def save_model(model, file_path:str) -> None:
    """
    Save a trained model to a file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug(f"Model saved to {file_path}")
    except FileNotFoundError as e:
        logger.error(f"Error saving model: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while saving the model: {e}")
        raise

def main():
    try:
        params = {'n_estimators': 25, 'random_state': 2}
        train_data = load_data(r'data\processed\train_tfidf.csv')
        X_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train, params)
        save_model(clf, r'models\model.pkl')
    except Exception as e:
        logger.error(f"Failed to complete the model building proces: {e}")
        print(f'Error: {e}')

if __name__ == "__main__":
    main()

