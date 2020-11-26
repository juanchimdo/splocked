from tensorflow.keras.preprocessing.sequence import pad_sequences

def convert_sentences(X):
    '''
    Convert to list of lists of word
    '''
    return [sentence.split(' ') for sentence in X]

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
    sentences = pad_sequences(X_token_train, dtype='float32', padding='post')
    return sentences
