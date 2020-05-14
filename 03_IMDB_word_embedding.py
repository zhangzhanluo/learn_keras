import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from matplotlib import pyplot as plt


def load_embedding_file(filepath, encoding='utf8', dtype='float32'):
    words2embedding = {}
    with open(filepath, encoding=encoding) as f:
        first = f.readline().strip().split()
        if len(first) > 2:
            vec = np.asarray(first[1:], dtype=dtype)
            words2embedding[first[0]] = vec
            dim = vec.size
        else:
            cnt, dim = first
        for line in f:
            c = line.strip().split()
            words2embedding[c[0]] = np.asarray(c[1:], dtype=dtype)
        return words2embedding, len(words2embedding), int(dim)


def load_embedding_weight(filepath, vocab2id, encoding='utf8', unknown=0):
    embedding_dict, cnt, dim = load_embedding_file(filepath, encoding)
    # '+1' for padding?
    weights = np.zeros((len(vocab2id) + 1, dim))
    for w, i in vocab2id.items():
        weights[i] = embedding_dict.get(w, unknown)
    return weights, dim


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
word_index = imdb.get_word_index()

emb_weights, emb_dim = load_embedding_weight('glove.6B.100d.txt', word_index)

vocab_size = len(word_index) + 1
max_len = 500
batch_size = 128

x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

model = Sequential()

# the embedding layer is here because of the difference between the word and the sentence
emb_layer = layers.Embedding(vocab_size, emb_dim, weights=[emb_weights], input_length=max_len)
model.add(emb_layer)
model.add(layers.Dropout(0.25))
model.add(layers.Conv1D(128, 1, padding='same', activation='relu', kernel_initializer='he_normal'))
model.add(layers.MaxPool1D(11, 3))
model.add(layers.Bidirectional(layers.GRU(128, return_sequences=False), merge_mode='sum'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.7))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile('rmsprop', loss='binary_crossentropy', metrics=['acc'])
epochs = 20
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
