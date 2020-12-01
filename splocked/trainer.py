import pandas as pd
import numpy as np
import joblib
import time
import datetime
import os

#from sklearn import linear_model
#from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from google.cloud import storage

from splocked.utils import *
from  splocked.model import *

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
BUCKET_SAMPLE_DATA_PATH = 'data/small_df.json'

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
    df = pd.read_json(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}", lines=True)
    if nrows:
      df = df.loc[:nrows]
    return df

def get_small_df(nrows=None):
    '''
    Loads a small json file that is 100 lines long
    with 30 spoilers and 40 non spoilers
    '''
    client = storage.Client()
    df = pd.read_json(f"gs://{BUCKET_NAME}/{BUCKET_SAMPLE_DATA_PATH}", lines=True)
    if nrows:
      df = df.loc[:nrows]
    return df

def preprocess(df, test_size=0.3):
    """method that pre-processes the data"""

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(df[['review']], np.array(df['is_spoiler']), test_size=test_size)

    # Make a word_to_id dict based on the training data
    word_dict = word_to_id(X_train)

    # Do a keras embedding based on the X_train word_to_id dictionary
    X_train = embed(X_train, word_dict)
    X_test = embed(X_test, word_dict)

    return X_train, X_test, y_train, y_test, word_dict


def word_to_id(X_train):
  pass

def preprocessing(list_of_sentences, word_to_id)

def train_model(X_train, y_train, vocab_size):
    """method that trains the model"""
    es = EarlyStopping(patience=5, restore_best_weights=True)

    model = init_model(vocab_size)
    model.fit(X_train, y_train,
          epochs=1,
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
    print("STARTING TO SAVE MODEL LOCALLY...", end='\n')
    model.save('models', save_format='tf')
    print("Saved model under models/saved_model.pb")

    print("STARTING TO SAVE MODEL IN GOOGLE CLOUD...", end='\n')
    model.save(f"gs://{BUCKET_NAME}/models/{MODEL_NAME}/{MODEL_VERSION}", save_format='tf')
    print(f"Saved model under gs://{BUCKET_NAME}/models/{MODEL_NAME}/{MODEL_VERSION}")

def evaluate(model, X_test, y_test):
    model.evaluate(X_test, y_test)

if __name__ == '__main__':
    # starting time
    start = time.time()
    #df = get_data(nrows=1_000)
    df = get_small_df()
    end = time.time()
    print(f'Runtime to get data from cloud: {end-start}')

    X_train, X_test, y_train, y_test, word_to_id = preprocess(df)
    print('Preprocessing the data')

    print('Training the model')
    model = train_model(X_train, y_train, len(word_to_id))

    print(f'The length of X_train is {len(X_train)}')
    print(f'The length of X_test is {len(X_test)}')

    print('Evaluating the model')
    res = model.evaluate(X_test, y_test)
    print(res)


    #'Saving Model'
    save_model(model)

    # Save Word to Dict
    # save_word_dict()

