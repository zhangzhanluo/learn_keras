import numpy as np
from keras.datasets import imdb
from keras import layers
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from matplotlib import pyplot as plt

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
word_index = imdb.get_word_index()

max_len = 500
batch_size = 128

x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

model = Sequential()


def multi_onehot(data, dim=10000):
    matrix = np.zeros((len(data), dim))
    for i, d in enumerate(data):
        # use a list to loc of a np array
        matrix[i, d] = 1
    return matrix


x_train = multi_onehot(x_train)
x_test = multi_onehot(x_test)

model.add(layers.Dense(128, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.7))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

epochs = 10
model.compile('adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=epochs, verbose=2)
train_acc, val_acc = history.history['acc'], history.history['val_acc']
train_loss, val_loss = history.history['loss'], history.history['val_loss']

plt.plot(range(1, epochs+1), train_acc, 'r-',
         range(1, epochs+1), val_acc, 'bo')
plt.legend(['Train Acc', 'Val Acc'])
plt.show()

plt.plot(range(1, epochs+1), train_loss, 'r-',
         range(1, epochs+1), val_loss, 'bo')
plt.legend(['Train Loss', 'Val Loss'])
plt.show()

