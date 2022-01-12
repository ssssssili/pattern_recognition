import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


def load_data(file_name):
    input_file = pd.read_csv(file_name)
    exoplanet_headers = input_file.columns.values
    exoplanet_data = input_file.values

    return exoplanet_headers, exoplanet_data


headers, data = load_data("all_exoplanets_2021.csv")

# print(headers)
# print(data)

print(data.shape)

model = models.Sequential()

# Example of adding layers to a model
# Make sure the shape of feature data is the same as the expected input, or else it won't work
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


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




