import pandas as pd
from splocked.utils import *
from  splocked.model  import *
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import datetime
import os
from google.cloud import storage
import pandas as pd
from sklearn import linear_model
import numpy as np
import joblib
import time
from tensorflow.keras.callbacks import EarlyStopping


### GCP configuration - - - - - - - - - - - - - - - - - - -

# /!\ you should fill these according to your account

### GCP Project - - - - - - - - - - - - - - - - - - - - - -

# not required here

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'splocked-betancourt-1'
CLOUD_PROJECT = 'splocked'


##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
# /!\ here you need to decide if you are going to train using the provided and uploaded data/train_1k.csv sample file
# or if you want to use the full dataset (you need need to upload it first of course)
BUCKET_TRAIN_DATA_PATH = 'data/IMDB_reviews.json'

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# not required here

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'splocked_models'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

# not required here

### - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def get_data(nrows=None):
    """method used in order to get the training data (or a portion of it) from google cloud bucket"""
    client = storage.Client()
    df = pd.read_json(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}", lines=True, nrows=nrows)
    return df

def preprocess(df, test_size = 0.3):
    """method that pre-processes the data"""
    # Selects only the specified amount of samples

    # Create a single column from the title of the review_summary and review_text as review
    df['review'] = df['review_summary'] + ' ' + df['review_text']

    # Filter only 'is_spoiler' and 'review' columns
    df = df[['is_spoiler', 'review']]

    # Convert boolean to binary the 'is_spoiler' function
    df['is_spoiler'] = boolean_to_binary_array(df['is_spoiler'])

    # Split the data and convert is_spoiler to np.array
    X_train, X_test, y_train, y_test = train_test_split(df['review'], np.array(df['is_spoiler']), test_size=test_size)

    # Make a word_to_id dict based on the training data
    word_dict = word_to_id(X_train)

    # Do a keras embedding based on the X_train word_to_id dictionary
    X_train = embed(X_train, word_dict)
    X_test = embed(X_test, word_dict)

    return X_train, X_test, y_train, y_test, word_dict

def train_model(X_train, y_train, vocab_size):
    """method that trains the model"""
    es = EarlyStopping(patience=5, restore_best_weights=True)

    model = init_model(vocab_size)
    model.fit(X_train, y_train,
          epochs=10,
          batch_size=32,
          validation_split=0.3,
          callbacks=[es]
         )
    return model

def save_model(model):
    """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""
    #local_model_name = 'model.joblib'

    # saving the trained model to disk (which does not really make sense
    # if we are running this code on GCP, because then this file cannot be accessed once the code finished its execution)
    # joblib.dump(model, local_model_name)
    # print("saved model.joblib locally")

    # client = storage.Client().bucket(BUCKET_NAME)

    # storage_location = '{}/{}/{}/{}'.format(
    #     'models',
    #     MODEL_NAME,
    #     MODEL_VERSION,
    #     local_model_name)
    # blob = client.blob(storage_location)
    # blob.upload_from_filename(local_model_name)
    # print("uploaded model.joblib to gcp cloud storage under \n => {}".format(storage_location))
    model.save(f"gs://${BUCKET_NAME}/${MODEL_NAME}/${UPLOADED_FILE_NAME}", save_format='tf')

if __name__ == '__main__':
    # starting time
    start = time.time()
    df = get_data(nrows=1_000)
    end = time.time()
    print(f'Runtime to get data from cloud: {end-start}')

    X_train, X_test, y_train, y_test, word_to_id = preprocess(df)
    print('Preprocessing the data')

    print('Training the model')
    model = train_model(X_train, y_train, len(word_to_id))

    print(f'The length of X_train is {len(X_train)}')
    print(f'The length of X_test is {len(X_test)}')

    #'Saving Model'
    #save_model(model)

