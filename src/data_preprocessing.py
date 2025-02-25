import os
import logging
import string

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Ensure logs directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

#logging config
logger = logging.getLogger('pre_processing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'pre_processing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """
    This function transforms the text by converting it to lower case, tokenizing, removing stopwords & punctuation, and stemming.
    """
    ps = PorterStemmer()
    
    #Convert to lower case
    text = text.lower()
    #Tokenize the text
    text = nltk.word_tokenize(text)
    # Remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]
    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    # Stem the tokens
    text = [ps.stem(word) for word in text]
    return ' '.join(text)


def preprocess_df(df, text_column='text', target_column='target'):
    """
    This function preprocesses the given dataframe by encoding the target column, removing duplicates, and transforming the text column.
    """
    try:
        logger.debug('Start preprocessing the dataframe')
        # Encode the target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')

        # Remove duplicate records
        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicate records removed')

        # Apply the text transformation function to the text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed')
        return df
    
    except KeyError as e:
        logger.error(f"Column not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during text normalization {e}")
        raise

def main(text_column = 'text', target_column = 'target'):
    """
    This function is the main entry point of the script. It loads the raw data, preprocesses it , and saves the preprocessed data.
    """
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded properly')
        
        # Transform the data
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        # Store the data inside data/processed

        # train_processed_data.to_csv('./data/processed/train.csv', index=False)
        # test_processed_data.to_csv('./data/processed/test.csv', index=False)
        # --------------------------OR---------------------------------------------
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        logger.debug('Data saved properly')
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except pd.errors.EmptyDataError as e:
        logger.error(f"No data in file: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()