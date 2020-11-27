import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


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
    embeds sentences for keras
    '''
    sentences = convert_sentences(sentences)
    sentences = tokenize(sentences, word_to_id)
    sentences = pad_sequences(sentences, dtype='float32', padding='post')
    return sentences

def boolean_to_binary_array(list):
    '''
    Takes a list of boolean values and returns
    an array of 0 if False and 1 if True
    '''
    return np.array([1 if x else 0 for x in list])


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
