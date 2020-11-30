from splocked.utils import convert_sentences
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import json
from splocked.utils import tokenize

def clean (text):

    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ') # Remove Punctuation

    lowercased = text.lower() # Lower Case

    tokenized = word_tokenize(lowercased) # Tokenize

    words_only = [word for word in tokenized if word.isalpha()] # Remove numbers

    stop_words = set(stopwords.words('english')) # Make stopword list

    without_stopwords = [word for word in words_only if not word in stop_words] # Remove Stop Words

    return " ".join(without_stopwords)

def stemm (text):

    tokenized = word_tokenize(text)

    stemmer = PorterStemmer()

    stemmed = [stemmer.stem(word) for word in tokenized]

    return " ".join(stemmed)

def preprocess_review(X, word_to_id_dict):

    clean_review = clean(X)

    stemed_review = stemm(clean_review)

    sentence_converted = convert_sentences([stemed_review])

    prediction_token = tokenize(sentence_converted, word_to_id_dict)

    prediction_pad = pad_sequences(prediction_token, maxlen=150, dtype='float32', padding='post')

    return prediction_pad
