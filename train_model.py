import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Define the directory
directory = 'models'

# Check if the directory exists, if not, create it
if not os.path.exists(directory):
    os.makedirs(directory)


# Load preprocessed data
data = np.load('data/preprocessed_data.npz')
X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

# Define data augmentation strategy
# This will improve the existing overfitting issue due to less data.
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the data generator on your training data
datagen.fit(X_train)

# Use the CNN model(sequential to train the data)
model = Sequential([
    tf.keras.layers.Input(shape=(150, 150, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.5),  # Dropout layer added here
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Dropout layer added here
    Dense(1, activation='sigmoid')
])

# Use an adam optimizer to check the loss and accuracy rate 
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model using the augmented data
model.fit(datagen.flow(X_train, y_train, batch_size=2), epochs=10, validation_data=(X_test, y_test))

# Save the model
model.save(os.path.join(directory, 'parking_space_model.keras'))