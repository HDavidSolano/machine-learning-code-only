from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# First, we need to normalize the data (always a good practice for regressions)

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std


# It will be evaluated multiple times, only contributing for a single variable each
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model    


# We will implement k-fold cross-validation

k = 4
num_val_samples = len(train_data) // k # integer division (neat)
num_epochs = 100
all_mae_histories = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples] # The validation data is a homogeneous chunk of the data
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    # The rest of the data is used to train each  instance of the model:
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],train_data[(i + 1) * num_val_samples:]], axis=0) 
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],train_targets[(i + 1) * num_val_samples:]],axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,validation_data=(val_data, val_targets),epochs=num_epochs, batch_size=1, verbose=0) # Record results
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)
    
# Accumulate histories to evaluate performance
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]


# Plot history
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# Now lets train the final model
model = build_model()
model.fit(train_data, train_targets, epochs=40, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(test_mae_score)

# -------Now I will proceed to save the model in different ways:-----------

# --- Save weights + architecture:
# Save the weights
model.save_weights('bh_weights.h5')
# Save the model architecture
with open('model_bh_architecture.json', 'w') as f:
    f.write(model.to_json())
    
    
#--- To load model + Architecture:    
'''
from keras.models import model_from_json
# Model reconstruction from JSON file
with open('model_bh_architecture.json', 'r') as f:
    model = model_from_json(f.read())
# Load weights into the new model
model.load_weights('bh_weights.h5')    
'''

# --- Save entire model:
from keras.models import load_model
# Creates a HDF5 file 'my_model.h5'
model.save('my_bh_model.h5')
# Deletes the existing model
del model  
# --- To load entire model:
# Returns a compiled model identical to the previous one
model = load_model('my_bh_model.h5')



    