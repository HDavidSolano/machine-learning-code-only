from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np
from keras import optimizers # This to configure your optimizer
from keras import losses     # This is to configure your loss function in detail
from keras import metrics    # This is to configure your metrics in detail
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# Setting up data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Setting up labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# Organizing the last 10000 entries for validation purposes
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]


model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Simplest command
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# Adding customizable optimizer
# model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
# Adding optimizer, loss, and metric customization
model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])

history = model.fit(partial_x_train, partial_y_train, epochs=10,batch_size=512, validation_data=(x_val, y_val))

results = model.evaluate(x_test, y_test) # To show final numbers

# To thest the NN
# model.predict(x_test)

#Plot the results (loss)
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.figure(1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Plot the results (acc)
acc_values = history_dict['binary_accuracy']
val_acc_values = history_dict['val_binary_accuracy']
plt.figure(2)
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

