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
