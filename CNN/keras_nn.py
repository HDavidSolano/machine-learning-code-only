from keras.models import Sequential
import numpy as np
from keras.layers import Dense
import keras
import pickle

train_x,train_y,test_x,test_y = pickle.load( open( "email_set1.pickle", "rb" ) )

n_classes = 20

train_x = np.array(train_x)
train_y = np.array(train_y)

test_x = np.array(test_x)
test_y = np.array(test_y)

model = Sequential()

model.add(Dense(units=400, activation='relu', input_dim=len(train_x[0])))
model.add(keras.layers.core.Dropout(0.25))
model.add(Dense(units=400, activation='relu', input_dim=400))
model.add(Dense(units=400, activation='relu', input_dim=400))
model.add(Dense(units=n_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),metrics=['accuracy'])

model.fit(train_x, train_y, epochs=3, batch_size=200)

loss_and_metrics, acc = model.evaluate(test_x, test_y, batch_size=40, verbose=1)
print(str(acc))
a = 1