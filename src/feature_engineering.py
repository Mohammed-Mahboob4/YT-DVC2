import os
import logging

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure logs directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

#logging config
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
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
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

def apply_tfid(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int = 10000) -> tuple:
    """
    Apply TF-IDF transformation to the text data in the train and test dataframes.
    """
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)

        X_train = train_data['text'].values
        X_test = test_data['text'].values
        Y_train = train_data['target'].values
        Y_test = test_data['target'].values

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = Y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = Y_test

        logger.debug('TF-IDF applied and data transformed')
        return train_df, test_df
    except Exception as e:
        logger.error(f"An error occurred during BOW transformation: {e}")
        raise

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Save the dataframe to a CSV file.    
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug(f"Data saved to {file_path}")
    except Exception as e:
        logger.error(f"An error occurred during data saving: {e}")
        raise

def main():
    try:
        max_features = 50

        train_data = load_data(r'data\interim\train_processed.csv')
        test_data = load_data(r"data\interim\test_processed.csv")

        train_df, test_df = apply_tfid(train_data, test_data, max_features)

        save_data(train_df, os.path.join("./data", "processed", "train_tfidf.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_tfidf.csv"))

    except Exception as e:
        logger.error(f"An error occurred during the main function: {e}")
        raise

if __name__ =="__main__":
    main()