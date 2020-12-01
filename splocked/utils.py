import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import requests
from bs4 import BeautifulSoup as bsp

def convert_sentences(X):
    '''
    Convert to list of lists of word
    '''
    return [sentence.split(' ') for sentence in X]

def word_to_id(sentences):
    word_dict = {}
    iter_ = 1
    for sentence in sentences:
        for word in sentence.split():
            if word in word_dict:
                continue
            word_dict[word] = iter_
            iter_ += 1
    return word_dict

def tokenize(sentences, word_to_id):
    '''
    Tokenize sentences
    '''
    return [[word_to_id[_] for _ in s if _ in word_to_id] for s in sentences]

def embed(sentences, word_to_id):
    '''
    embeds sentences for keras model
    '''
    sentences = convert_sentences(sentences)
    sentences = tokenize(sentences, word_to_id)
    sentences = pad_sequences(sentences, dtype='float32', padding='post', maxlen=150)

    return sentences

def boolean_to_binary_array(list):
    '''
    Takes a list of boolean values and returns
    an array of 0 if False and 1 if True
    '''
    return np.array([1 if x else 0 for x in list])

def imdb_api(movie_title):
    '''
    This function returns the movie's imdb ID
    '''
    title = movie_title.strip(' ').replace(' ', '+').title()
    api_key = 'ca58a32b'
    api_url = 'http://www.omdbapi.com/'
    params = {'t':title, 'apikey':api_key}
    response = requests.get(api_url, params = params).json()
    imdbID = response['imdbID']
    return imdbID

def get_reviews(url):

    response = requests.get(url)
    soup = bsp(response.content, "html.parser")
    reviews = []
    for comment in soup.find_all("div", class_="lister-item-content"):
        titles = comment.find("a", class_="title").string.rstrip('\n').strip(' ')
        comments = comment.find_all("div", class_='text')
        for cmt in comments:
            reviews.append({'title':titles, 'comment': cmt.text})
    return pd.DataFrame(reviews)

if __name__ == "__main__":
    # A random review to convert
    sentences = [
        'Hello everybody!',
        'Have a great day',
        'This is a sentence'
      ]

    # Embed the sentences
    word_dict = word_to_id(sentences)
    print(embed(sentences, word_dict))

    # True and false list
    booleans = [True, False, True, False]
    print(boolean_to_binary_array(booleans))
