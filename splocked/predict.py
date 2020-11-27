from splocked.utils import convert_sentences
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_review(X):

    review_to_predict = [X]

    sentence_converted = convert_sentences(review_to_predict)

    prediction_token = tokenize(sentence_converted, word_to_id)

    prediction_pad = pad_sequences(prediction_token, maxlen=150, dtype='float32', padding='post')

    return prediction_pad
