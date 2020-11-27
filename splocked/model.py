from tensorflow.keras import Sequential
from tensorflow.keras import layers

def init_model(vocab_size):
    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size+1, output_dim=30, mask_zero=True))
    model.add(layers.LSTM(10))
    model.add(layers.Dense(5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model
