import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


def load_data(file_name):
    input_file = pd.read_csv(file_name)
    exoplanet_headers = input_file.columns.values
    exoplanet_data = input_file.values

    return exoplanet_headers, exoplanet_data


def train_test_split(headers, data, column, test_size):
    split = int(len(data) * test_size)

    column_index = np.where(headers == column)[0][0]

    # for checking the model with something more simplistic
    orbital_period_days_index = np.where(headers == "Orbital Period Days")[0][0]
    mass_index = np.where(headers == "Mass")[0][0]

    training_data = data[0: split]
    test_data = data[split: len(data)]

    x_train = training_data[:, orbital_period_days_index:mass_index]
    x_test = test_data[:, orbital_period_days_index:mass_index]

    y_train = training_data[:, column_index]
    y_test = test_data[:, column_index]
    """
    
    x_test = x_train[:, column_index]
    y_test = y_train[:, column_index]
    """

    # If random: randomize

    return np.asarray(x_train).astype('float32'),\
           np.asarray(x_test).astype('float32'),\
           np.asarray(y_train).astype('float32'),\
           np.asarray(y_test).astype('float32')


headers, data = load_data("all_exoplanets_2021.csv")

x_train, x_test, y_train, y_test = train_test_split(headers, data, 'Stellar Effective Temperature', 0.25)

x_train = x_train.reshape(list(x_train.shape) + [1])
x_train = x_train.reshape(list(x_train.shape) + [1])
y_train = y_train.reshape(list(y_train.shape) + [1])
y_train = y_train.reshape(list(y_train.shape) + [1])
print(x_train.shape)
print(y_train.shape)
model = models.Sequential()

# Example of adding layers to a model
# Make sure the shape of feature data is the same as the expected input, or else it won't work

# TODO: Try to get the shape right to fit
model.add(layers.Conv2D(32, (1, 1), activation='relu', input_shape=x_train.shape[1:]))
"""
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
"""
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x_train, y_train)
exit()

# Below is an example setup, we can tweak this

    # PREPROCESSING
def filtering(full_dataset):    # Filter out all the rows that are missing data (and thus can't be used for training)
    ...
    return filtered_dataset

def divide_dataset(complete_dataset):   #takes a (complete) dataset and splits it into two parts (one for training, one for validation)
    ...
    return training_set, validation_set

def make_incomplete_set(complete_dataset):     # Remove some datapoints from a (complete) dataset to make a training dataset
    ... 
    return incomplete_dataset 

def preprocessing(dataset):
    complete_dataset = filtering(dataset)
    complete_training_set, complete_validation_set = divide_dataset(complete_dataset)
    incomplete_training_set = make_incomplete_set(complete_training_set)
    incomplete_validation_set = make_incomplete_set(complete_validation_set)
    return complete_training_set, incomplete_training_set, complete_validation_set, incomplete_validation_set

    # MODEL
def fill_in_blanks(incomplete_data): # Fill in some average values for missing data
    ...
    return estimated_data

def is_complete(data):   #checks if there are values missing in the data, returning False if values are missing or True if the data is complete
    ...
    return True/False

def not_converged(data1, data2):    # Checks if the difference between the two data vectors is smaller than some threshold (keep in mind to normalize values!)
    ...
    return True/False

def apply_model(input_data):
    if is_complete(input_data):
        return input_data
    else:
        ...
        old_data = fill_in_blanks(input_data)
        new_data = old_data
        while not_converged(old_data, new_data):
            old_data = new_data
            new_data = 0.5*(new_data + tf.ourmodel.applymodel(new_data))  # We'll obviously need to use different functions for this, this is more of an example of what it should do
    return new_data 
    # TRAINING
# honestly, I'm not quite sure what the tf methods for training a model that is called inside a function is




