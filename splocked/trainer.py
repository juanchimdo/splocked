import pandas as pd
import numpy as np
#import joblib
import time
import datetime
import os
import json

from sklearn import linear_model
from sklearn.model_selection import train_test_split
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
BUCKET_CLEANED_DATA_PATH = 'data/data_cleaned.csv'
LOCAL_FOLDER_NAME = 'google_cloud_model'

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# not required here

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'google_cloud_model'

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

def preprocessing(list_of_sentences, word_to_id):
  pass

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
    model.save(LOCAL_FOLDER_NAME, save_format='tf')
    print(f"Saved model under {LOCAL_FOLDER_NAME}/saved_model.pb")

    print("STARTING TO SAVE MODEL IN GOOGLE CLOUD...", end='\n')
    model.save(f"gs://{BUCKET_NAME}/models/{MODEL_NAME}/{MODEL_VERSION}", save_format='tf')
    print(f"Saved model under gs://{BUCKET_NAME}/models/{MODEL_NAME}/{MODEL_VERSION}")

def evaluate(model, X_test, y_test):
    model.evaluate(X_test, y_test)

if __name__ == '__main__':
    # starting time
    start = time.time()
    # Get the data that is already cleaned
    data = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_CLEANED_DATA_PATH}")
    #data = pd.read_csv('raw_data/data_cleaned.csv')
    # Shuffle the data
    df_shuffle = data.sample(frac=1).copy()
    df_shuffle.reset_index(inplace =True)
    # Drop the index
    df_shuffle.drop(columns='index', inplace= True)
    end = time.time()
    print(f'Runtime to get cleaned and shuffled data: {end-start}')

    # Split the data into train and test
    print('Splitting into balanced and unbalanced datasets')

    df_shuffle_test = df_shuffle.loc[:200_000]
    df_shuffle_train = df_shuffle.loc[200_000:]

    # Balance the training samples
    print('Balancing the training samples')
    g = df_shuffle_train.groupby('is_spoiler')
    g = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
    g = g.set_index('is_spoiler')
    g = g.reset_index()
    df_shuffle_train = g

    # Get 100,000 samples
    # n = 100
    # print(f'Selecting only the first {n} samples')
    # df_sample_train = df_shuffle_train.sample(n=n)

    print("Selecting ALL the samples")
    df_sample_train = df_shuffle_train

    # Define  X and y
    print("Processing the data")
    X = df_sample_train[['clean_reviews']]
    y = df_sample_train['is_spoiler']

    # Split the train sample into train/test again
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    # Convert sentences into list of words
    X_train = X_train.apply(convert_sentences)
    X_test = X_test.apply(convert_sentences)

    # Make word_to_id_dictionary
    word_to_id = {}
    iter_ = 1
    for sentence in X_train['clean_reviews']:
        for word in sentence:
            if word in word_to_id:
                continue
            word_to_id[word] = iter_
            iter_ += 1

    # Tokenize X
    X_token_train = tokenize(X_train['clean_reviews'], word_to_id)
    X_token_test = tokenize(X_test['clean_reviews'], word_to_id)

    # Add padding
    X_train_maxlen = pad_sequences(X_token_train, maxlen=250, dtype='float32', padding='post')
    X_test_maxlen = pad_sequences(X_token_test, maxlen=250, dtype='float32', padding='post')
    print("Finished processing the data")

    # Train model
    print("Train the model")
    model = init_model(len(word_to_id))
    es = EarlyStopping(patience=7, restore_best_weights=True)
    history = model.fit(X_train_maxlen, y_train, epochs=25, batch_size=16, validation_split=0.2, callbacks=[es])

    with open(f"{LOCAL_FOLDER_NAME}/history.json", "w") as hist:
      json.dump(history.history, hist)

    # Evaluate model of the train/test split
    print("Starting to evaluate model on balanced data")
    res_bal = model.evaluate(X_test_maxlen, y_test)
    print(" RESULTS FOR BALANCED DATA")
    print(f'Loss:{res_bal[0]}')
    print(f'Recall:{res_bal[1]}')

    # Evalute model on the true balanced data
    X_shuffle_test = df_shuffle_test[['clean_reviews']]
    y_shuffle_test = df_shuffle_test['is_spoiler']

    X_shuffle_test_converted = X_shuffle_test.apply(convert_sentences)
    X_shuffle_test_tokenized = tokenize(X_shuffle_test_converted['clean_reviews'], word_to_id)
    X_shuffle_test_maxlen = pad_sequences(X_shuffle_test_tokenized, maxlen=250, dtype='float32', padding='post')

    print("Starting to evaluate model on tru balance data")
    res_true = model.evaluate(X_shuffle_test_maxlen, y_shuffle_test)
    print(" RESULTS FOR TRUE BALANCE DATA")
    print(f'Loss:{res_bal[0]}')
    print(f'Recall:{res_bal[1]}')

    #'Saving Model'
    print("Saving model")
    save_model(model)

    # Save Word to Dict
    print("Saving word_to_id")
    with open(f"{LOCAL_FOLDER_NAME}/word_to_id.json", 'w') as fp:
        json.dump(word_to_id, fp)
