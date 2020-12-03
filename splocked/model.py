from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.metrics import Recall
#from tensorflow_addons.metrics import F1Score

recall = Recall(name='recall')
#f1 = F1Score(name='f1_score', num_classes=1)

def init_model(vocab_size):
    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size+1, output_dim=30, mask_zero=True))
    model.add(layers.GRU(units=128 , recurrent_dropout = 0.1 , dropout = 0.1))
    #model.add(layers.GRU(units=128 , recurrent_dropout = 0.1 , dropout = 0.1, return_sequences=True, input_shape=(X_train_maxlen.shape[1],200)))
    #model.add(layers.GRU(units=64, return_sequences=True, input_shape=(X_train_maxlen.shape[1],150)))
    #model.add(layers.GRU(units=32))
    #model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=recall)

    return model
