import pandas as pd
from splocker.utils import boolean_to_binary_array, word_to_id, embed

def get_data():
    """method used in order to get the training data (or a portion of it) from google cloud bucket"""
    df = pd.read_json('./../../raw_data/IMDB_reviews.json', lines=True)
    return df

def preprocess(df, test_size = 0.3, ):
    """method that pre-processes the data"""
    # Create a single column from the title of the review_summary and review_text as review
    df['review'] = df['review_summary'] + ' ' + df['review_text']

    # Filter only 'is_spoiler' and 'review' columns
    df = df[['is_spoiler', 'review']]

    # Convert boolean to binary the 'is_spoiler' function
    df['is_spoiler'] = boolean_to_binary_array(df['is_spoiler'])

    # Split the data and convert is_spoiler to np.array
    X_train, X_test, y_train, y_test = train_test_split(small_df['review'], np.array(small_df['is_spoiler']), test_size=test_size)

    # Make a word_to_id dict based on the training data
    word_dict = word_to_id(X_train)

    # Do a keras embedding based on the X_train word_to_id dictionary
    X_train = embed(X_train, word_dict)
    X_test = embed(X_test, word_dict)

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """method that trains the model"""
    rgs = linear_model.Lasso(alpha=0.1)
    rgs.fit(X_train, y_train)
    return rgs


def save_model(reg):
    """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""

    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    # Implement here
    print("saved model.joblib locally")

    # Implement here
    print("uploaded model.joblib to gcp cloud storage under \n => {}".format(storage_location))

if __name__ == '__main__':
df = get_data()
X_train, y_train = preprocess(df)
clf = train_model(X_train, y_train)
save_model(clf)


